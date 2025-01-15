import torch
import numpy as np
import spacy
from spacy.matcher import Matcher
from transformers import BertTokenizer, BertModel

#####################################################
# 1) Load models globally (SpaCy + BERT) at import
#####################################################
nlp = spacy.load("en_core_web_sm")

# You can change to any BERT-like model, e.g. 'distilbert-base-uncased'
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # we only do inference for embeddings

CLOSED_LOOPS_VALUE = 10

def parse_graph_description(text):
    """
    Extract numeric info from the graph description using spaCy + patterns.
    Returns a dict:
      - n_nodes
      - n_edges
      - average_degree
      - n_triangles
      - clustering_coefficient
      - max_kcore
      - n_communities
      - triangles_description
    """
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Patterns as in your original code (omitted here for brevity),
    # but you can copy them directly from your snippet.
    # -------------- (Same as your existing patterns) ----------------
    pattern_nodes = [
        {"IS_DIGIT": True}, 
        {"LOWER": {"IN": ["node", "nodes"]}}
    ]
    matcher.add("NODES", [pattern_nodes])

    pattern_edges = [
        {"IS_DIGIT": True},
        {"LOWER": {"IN": ["edge", "edges"]}}
    ]
    matcher.add("EDGES", [pattern_edges])

    pattern_avg_degree = [
        {"LOWER": "average"},
        {"LOWER": "degree"},
        {"LOWER": {"IN": ["is", "=", "equal"]}, "OP": "?"},
        {"LOWER": {"IN": ["is", "=", "equal"]}, "OP": "?"},
        {"LOWER": "to", "OP": "?"},
        {"LIKE_NUM": True, "OP": "+"}
    ]
    pattern_avg_degree2 = [
        {"LOWER": "on"},
        {"LOWER": "average"},
        {"IS_PUNCT": True, "OP": "*"},
        {"LOWER": "each"},
        {"LOWER": {"IN": ["node", "nodes"]}},
        {"LOWER": {"IN": ["is", "are"]}, "OP": "?"},
        {"LOWER": "connected"},
        {"LOWER": "to"},
        {"LIKE_NUM": True, "OP":"+"},
        {"LOWER": {"IN": ["other", "neighbors", "nodes"]}, "OP": "*"}
    ]
    matcher.add("AVERAGE_DEGREE", [pattern_avg_degree, pattern_avg_degree2])

    pattern_triangles_1 = [
        {"IS_DIGIT": True},
        {"LOWER": {"IN": ["triangle", "triangles"]}}
    ]
    pattern_triangles_2 = [
        {"LOWER": {"IN": ["there", "there're", "thereare"]}, "OP": "*"},
        {"LOWER": "are", "OP": "?"},
        {"IS_DIGIT": True},
        {"LOWER": {"IN": ["triangle", "triangles"]}}
    ]
    matcher.add("TRIANGLES", [pattern_triangles_1, pattern_triangles_2])

    pattern_clust = [
        {"LOWER": "global"},
        {"LOWER": "clustering"},
        {"LOWER": "coefficient"},
        {"LOWER": {"IN": ["is", "=", "are"]}, "OP": "?"},
        {"LIKE_NUM": True, "OP": "+"}
    ]
    matcher.add("CLUSTERING", [pattern_clust])

    pattern_clust_kcore = [
        {"LOWER": "global"},
        {"LOWER": "clustering"},
        {"LOWER": "coefficient"},
        {"IS_PUNCT": True, "OP": "*"}, 
        {"LOWER": "and"},
        {"LOWER": "the", "OP": "?"},
        {"LOWER": "graph", "OP": "?"},
        {"LOWER": "'s", "OP": "?"},
        {"LOWER": {"IN": ["maximum", "max"]}},
        {"LOWER": "k"},
        {"LOWER": "-", "OP": "?"},
        {"LOWER": "core"},
        {"IS_PUNCT": True, "OP": "*"},
        {"LOWER": {"IN": ["are", "=", "is"]}},
        {"IS_PUNCT": True, "OP": "*"},
        {"LIKE_NUM": True, "OP": "+"},
        {"IS_PUNCT": True, "OP": "*"},
        {"LOWER": "and"},
        {"IS_PUNCT": True, "OP": "*"},
        {"LIKE_NUM": True, "OP": "+"},
        {"LOWER": "respectively", "OP": "?"},
        {"IS_PUNCT": True, "OP": "*"}
    ]
    matcher.add("CLUST_KCORE_RESPECTIVELY", [pattern_clust_kcore])

    pattern_kcore_1 = [
        {"LOWER": {"IN": ["maximum", "max"]}},
        {"LOWER": "k"},
        {"LOWER": "-"},
        {"LOWER": "core"},
        {"LOWER": {"IN": ["of", "is", "=", "are"]}, "OP": "*"},
        {"LIKE_NUM": True, "OP": "+"}
    ]
    matcher.add("KCORE", [pattern_kcore_1])

    pattern_communities_1 = [
        {"LOWER": {"IN": ["community", "communities"]}},
        {"LOWER": {"IN": ["=", "equal", "equals", "is", "are", "to"]}, "OP": "*"},
        {"IS_DIGIT": True, "OP": "+"}
    ]
    pattern_communities_2 = [
        {"LOWER": {"IN": ["consists", "has"]}, "OP": "?"},
        {"LOWER": "of", "OP": "?"},
        {"IS_DIGIT": True, "OP": "+"},
        {"LOWER": {"IN": ["community", "communities"]}}
    ]
    matcher.add("COMMUNITIES", [pattern_communities_1, pattern_communities_2])

    pattern_triangles_desc = [
        {"LOWER": "forming"},
        {"LOWER": "closed"},
        {"LOWER": "loops"},
        {"LOWER": "of"},
        {"LOWER": "nodes"}
    ]
    matcher.add("TRIANGLES_DESC", [pattern_triangles_desc])

    info = {
        "n_nodes": 0,
        "n_edges": 0,
        "average_degree": 0,
        "n_triangles": 0,
        "clustering_coefficient": 0,
        "max_kcore": 0,
        "n_communities": 0,
        "triangles_description": 0
    }

    matches = matcher(doc)

    def tokens_to_float(tokens):
        text_concat = "".join(t.text for t in tokens)
        try:
            return float(text_concat)
        except ValueError:
            return 0

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]

        if rule_id == "CLUST_KCORE_RESPECTIVELY":
            digit_tokens = [t for t in span if t.like_num]
            if len(digit_tokens) >= 2:
                clust_val = tokens_to_float([digit_tokens[0]])
                kcore_val = tokens_to_float([digit_tokens[1]])
                info["clustering_coefficient"] = clust_val
                info["max_kcore"] = int(kcore_val) if kcore_val is not None else 0
            continue

        digit_tokens = [t for t in span if t.like_num]
        if not digit_tokens:
            if rule_id == "TRIANGLES_DESC":
                info["triangles_description"] = CLOSED_LOOPS_VALUE
            continue

        val = tokens_to_float(digit_tokens)

        if rule_id == "NODES":
            info["n_nodes"] = int(val) if val is not None else 0
        elif rule_id == "EDGES":
            info["n_edges"] = int(val) if val is not None else 0
        elif rule_id == "AVERAGE_DEGREE":
            info["average_degree"] = val
        elif rule_id == "TRIANGLES":
            info["n_triangles"] = int(val) if val is not None else 0
        elif rule_id == "CLUSTERING":
            info["clustering_coefficient"] = val
        elif rule_id == "KCORE":
            info["max_kcore"] = int(val) if val is not None else 0
        elif rule_id == "COMMUNITIES":
            info["n_communities"] = int(val) if val is not None else 0

    return info

def get_spacy_numeric_feats(text: str) -> np.ndarray:
    """
    Returns an 8D numpy array of numeric features from the text.
    Order is:
    [n_nodes, n_edges, average_degree, n_triangles,
     clustering_coefficient, max_kcore, n_communities,
     triangles_description]
    """
    info_dict = parse_graph_description(text)
    # Convert dict to array in a stable order
    return np.array(list(info_dict.values()), dtype=np.float32)

def get_bert_embedding(text: str, device="cpu") -> torch.Tensor:
    """
    Returns a single [1, 768] embedding for the textual description using BERT.
    By default, we take the [CLS] token's hidden state. 
    """
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        # Move input to device if you want to run on GPU
        encoded = {k: v.to(device) for k,v in encoded.items()}
        outputs = bert_model(**encoded)
    # outputs.last_hidden_state shape: [1, seq_len, hidden_size]
    # We take the [CLS] token at index 0
    cls_emb = outputs.last_hidden_state[:, 0, :]  # shape [1, 768]
    return cls_emb.detach().cpu()  # Return to CPU if needed

def get_combined_feats(text: str, device="cpu", embed_dim=768) -> torch.Tensor:
    """
    1) Extract numeric features with spaCy
    2) Extract BERT embedding
    3) Concatenate into a single vector: [1, 8 + 768]
    """
    numeric_feats = get_spacy_numeric_feats(text)  # shape (8,)
    numeric_tensor = torch.from_numpy(numeric_feats).unsqueeze(0).float()  # [1, 8]

    bert_emb = get_bert_embedding(text, device=device)  # [1, 768]
    combined = torch.cat([numeric_tensor, bert_emb], dim=1)  # [1, 8 + 768]
    return combined  # shape [1, 776]

def example_debug():
    text = (
      "In this graph, there are 46 nodes connected by 460 edges. "
      "On average, each node is connected to 20.0 other nodes. "
      "Within the graph, there are 1280 triangles, forming closed loops of nodes. "
      "The global clustering coefficient is 0.4294821608321217. "
      "Additionally, the graph has a maximum k-core of 16 and a number of communities equal to 5."
    )
    combined = get_combined_feats(text)
    print("Combined feat shape:", combined.shape)  # [1, 776]
    print("Combined feat example:", combined)

if __name__ == "__main__":
    example_debug()
