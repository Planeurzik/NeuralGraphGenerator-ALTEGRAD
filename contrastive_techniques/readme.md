# Contrastive Techniques

In this iteration, we incorporated **contrastive learning** techniques to enhance the model's ability to learn robust graph embeddings. By creating augmented views of each graph and maximizing the agreement between their embeddings, the model achieves better generalization and improved performance in graph generation tasks.

### Contrastive Learning Components:

- **Graph Augmentation (`graph_augment`):** Applies random edge dropping to create different views of the same graph.
- **Projection Head (`Projector`):** Maps encoder outputs to a lower-dimensional space suitable for contrastive loss computation.
- **NT-Xent Loss (`nt_xent_loss`):** Encourages similar embeddings for augmented views while distinguishing between different graphs.

These enhancements aim to improve the quality and realism of the generated graphs, making the model more effective in applications requiring precise structural properties.

The main changes are in `utils.py`