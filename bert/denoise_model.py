import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
#                               Helper Functions
##############################################################################

def extract(a, t, x_shape):
    """
    Given a 1D tensor 'a' (e.g. betas) of length timesteps,
    select the index t[i] for each sample in the batch.
    Then reshape so it can broadcast over 'x_shape'.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    Forward diffusion step: q(x_t | x_0).
    x_start: the original (latent) data x_0
    t:       a 1D tensor [batch_size], each entry in [0, timesteps-1]
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # x_t = sqrt( alpha_t ) * x_0 + sqrt(1 - alpha_t) * noise
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, cond,
             sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
             noise=None, loss_type="l1"):
    """
    Compute the denoising loss at time t.
    denoise_model predicts the noise given x_noisy, t, and cond.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # Forward diffusion to get x_noisy
    x_noisy = q_sample(x_start, t,
                       sqrt_alphas_cumprod,
                       sqrt_one_minus_alphas_cumprod,
                       noise=noise)

    # Predict the noise
    predicted_noise = denoise_model(x_noisy, t, cond)

    # Compare predicted noise with the true noise
    if loss_type == 'l1':
        loss = F.l1_loss(predicted_noise, noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(predicted_noise, noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(predicted_noise, noise)
    else:
        raise NotImplementedError(f"Unknown loss_type: {loss_type}")

    return loss

##############################################################################
#                           Time Embedding Module
##############################################################################

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encode time step 't' into a high-dimensional vector using sin/cos embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: [batch_size], integer time steps
        Return: [batch_size, dim], the positional embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)

        # [half_dim]
        freq = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # [batch_size, half_dim]
        theta = time[:, None].float() * freq[None, :]

        # Final shape: [batch_size, 2*half_dim]
        embeddings = torch.cat((theta.sin(), theta.cos()), dim=-1)
        return embeddings

##############################################################################
#                           Refined Denoise Model
##############################################################################

class DenoiseNN(nn.Module):
    """
    A simpler approach that:
      1) Embeds 'cond' (the condition vector) once.
      2) Embeds time steps via sinusoidal embedding, merges them.
      3) Feeds [x, cond_emb, time_emb] into MLP, producing predicted noise.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        """
        Args:
            input_dim: dimension of x (the latent vector)
            hidden_dim: dimension of hidden layers
            n_layers: number of feed-forward layers
            n_cond: dimension of condition input
            d_cond: dimension to which we map the condition
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_cond   = n_cond
        self.d_cond   = d_cond

        # Condition MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Let's define an MLP:
        # First layer: input_dim + d_cond + hidden_dim (time) -> hidden_dim
        in_dim_first = input_dim + d_cond + hidden_dim
        self.fc_in = nn.Linear(in_dim_first, hidden_dim)

        # Middle layers
        self.mid_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.mid_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Final layer outputs [batch_size, input_dim]
        self.fc_out = nn.Linear(hidden_dim, input_dim)

        # We'll apply a simple ReLU and optional BatchNorm
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)])
        self.act = nn.ReLU()

    def forward(self, x, t, cond):
        """
        x: [batch_size, input_dim]
        t: [batch_size], integer time steps
        cond: [batch_size, n_cond], the condition vector
        Returns: predicted_noise, shape = [batch_size, input_dim]
        """
        # Clean up cond: handle NaNs
        cond = cond.view(-1, self.n_cond)
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond_emb = self.cond_mlp(cond)   # [bs, d_cond]

        # Time embedding => [bs, hidden_dim]
        t_emb = self.time_mlp(t)

        # Concatenate once
        cat_input = torch.cat([x, cond_emb, t_emb], dim=1)  # [bs, input_dim + d_cond + hidden_dim]

        # Pass first layer
        h = self.fc_in(cat_input)
        h = self.act(h)
        h = self.bns[0](h)

        # Middle layers
        for i, layer in enumerate(self.mid_layers, start=1):
            h = layer(h)
            h = self.act(h)
            h = self.bns[i](h)

        # Final
        out = self.fc_out(h)  # [bs, input_dim]
        return out

##############################################################################
#                            Improved Decoder
##############################################################################

class Decoder(nn.Module):
    """
    A decoder that transforms a latent vector z of shape (batch_size, latent_dim)
    into an adjacency matrix of shape (batch_size, n_nodes, n_nodes).
    - We only produce the upper triangle of adjacency and mirror it to ensure symmetry.
    - We apply a Sigmoid to get a continuous probability. Then you can threshold
      at inference if you want a discrete 0/1 adjacency.
    """
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = (n_nodes * (n_nodes - 1)) // 2  # number of edges in upper triangle

        # Build an MLP
        layers = []
        in_dim = latent_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Final layer => produce logits for each edge in the upper triangle
        layers.append(nn.Linear(in_dim, self.n_edges))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        """
        z: [batch_size, latent_dim]
        Returns: adjacency: [batch_size, n_nodes, n_nodes], in [0,1]
        """
        # Predict logits
        logits = self.mlp(z)  # [batch_size, n_edges]
        # Convert to probabilities
        edge_probs = torch.sigmoid(logits)  # [batch_size, n_edges]

        # Build adjacency
        batch_size = z.size(0)
        adj = torch.zeros(batch_size, self.n_nodes, self.n_nodes, device=z.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)  # (2, n_edges)
        adj[:, idx[0], idx[1]] = edge_probs
        # Mirror the upper triangle
        adj = adj + adj.transpose(1, 2)
        return adj

##############################################################################
#                              Sampling Routines
##############################################################################

@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    """
    One reverse diffusion step to predict x_{t-1} from x_t.
    model: denoise_model
    x: [batch_size, latent_dim] at time t
    t:  [batch_size] of time indices
    cond: [batch_size, n_cond]
    betas: 1D array of length timesteps
    t_index: integer used to check if t=0
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Predicted noise
    pred_noise = model(x, t, cond)
    # eq. (11) in DDPM: model_mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    """
    Start from pure noise, repeatedly apply p_sample to get final samples.
    shape: (batch_size, latent_dim)
    """
    device = next(model.parameters()).device
    b = shape[0]

    x = torch.randn(shape, device=device)
    imgs = []
    for i in reversed(range(timesteps)):
        x = p_sample(model, x,
                     t=torch.full((b,), i, device=device, dtype=torch.long),
                     cond=cond,
                     t_index=i,
                     betas=betas)
        imgs.append(x)
    return imgs

@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    """
    Main entry point to run the reverse diffusion process.
    Returns a list of length 'timesteps', each entry is [batch_size, latent_dim].
    The final entry in the list is x_0.
    """
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))

##############################################################################
# Example usage:
#
#   - DenoiseNN for generating latents
#   - Decoder to produce adjacency
#
#   During training, you would do something like:
#     1. latents = ...
#     2. compute p_losses(denoise_model, latents, t, cond, ...)
#     3. backprop
#
#   During inference:
#     1. samples = sample(denoise_model, cond, latent_dim, timesteps, betas, batch_size)
#     2. final_latent = samples[-1]
#     3. adjacency = decoder(final_latent)
##############################################################################
