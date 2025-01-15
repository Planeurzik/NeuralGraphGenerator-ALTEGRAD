import spacy
from spacy.matcher import Matcher
import numpy as np

CLOSED_LOOPS_VALUE = 10

# Charge le petit modèle anglais de spaCy
nlp = spacy.load("en_core_web_sm")


def parse_graph_description(text):
    """
    Extrait des informations clés depuis une description de graphe.
    Retourne un dictionnaire avec:
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
    """
    print("=== Debug tokens ===")
    for i, token in enumerate(doc):
        print(f"{i}: '{token.text}'  LOWER='{token.lower_}'  POS={token.pos_}  DEP={token.dep_}")
    print("====================\n")
    """

    matcher = Matcher(nlp.vocab)

    # --- 1) NODES : Pattern du type "[nombre] nodes"
    pattern_nodes = [
        {"IS_DIGIT": True}, 
        {"LOWER": {"IN": ["node", "nodes"]}}
    ]
    matcher.add("NODES", [pattern_nodes])

    # --- 2) EDGES : Pattern "[nombre] edges"
    pattern_edges = [
        {"IS_DIGIT": True},
        {"LOWER": {"IN": ["edge", "edges"]}}
    ]
    matcher.add("EDGES", [pattern_edges])

    # --- 3) AVERAGE DEGREE (forme simple) => "average degree is X"
    pattern_avg_degree = [
        {"LOWER": "average"},
        {"LOWER": "degree"},
        # On autorise (is|=|equal) OU rien
        {"LOWER": {"IN": ["is", "=", "equal"]}, "OP": "?"},
        {"LOWER": {"IN": ["is", "=", "equal"]}, "OP": "?"},
        # On autorise "to"
        {"LOWER": "to", "OP": "?"},
        {"LIKE_NUM": True, "OP": "+"}
    ]
    pattern_avg_degree2 = [
        {"LOWER": "on"},
        {"LOWER": "average"},
        {"IS_PUNCT": True, "OP": "*"},   # tolérer une virgule
        {"LOWER": "each"},
        {"LOWER": {"IN": ["node", "nodes"]}},
        {"LOWER": {"IN": ["is", "are"]}, "OP": "?"},
        {"LOWER": "connected"},
        {"LOWER": "to"},
        {"LIKE_NUM": True, "OP":"+"},   # ex: 10.0
        {"LOWER": {"IN": ["other", "neighbors", "nodes"]}, "OP": "*"}
    ]
    matcher.add("AVERAGE_DEGREE", [pattern_avg_degree, pattern_avg_degree2])

    # --- 4) TRIANGLES : "X triangles" ou "there are X triangles"
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

    # --- 5) GLOBAL CLUSTERING COEFFICIENT (forme simple)
    pattern_clust = [
        {"LOWER": "global"},
        {"LOWER": "clustering"},      # spaCy te renvoie ce token
        {"LOWER": "coefficient"},     # idem
        {"LOWER": {"IN": ["is", "=", "are"]}, "OP": "?"},
        {"LIKE_NUM": True, "OP": "+"} # ex: 0.345238...
    ]
    matcher.add("CLUSTERING", [pattern_clust])

    # --- 5 bis) CLUSTERING + KCORE dans la même phrase, ex: "are X and Y respectively"
    # ex: "The global clustering coefficient and the graph's maximum k-core are 0.8334 and 32 respectively."
    pattern_clust_kcore = [
        # 1) global clustering coefficient
        {"LOWER": "global"},
        {"LOWER": "clustering"},
        {"LOWER": "coefficient"},
        {"IS_PUNCT": True, "OP": "*"},   # ex: virgule éventuelle
        
        {"LOWER": "and"},
        
        # 2) the graph 's maximum k - core
        # Autorise "the", "graph", "'s" de manière optionnelle
        {"LOWER": "the", "OP": "?"},
        {"LOWER": "graph", "OP": "?"},
        {"LOWER": "'s", "OP": "?"},
        
        {"LOWER": {"IN": ["maximum", "max"]}},
        
        # ici spaCy sépare "k" "-" "core"
        {"LOWER": "k"},
        {"LOWER": "-", "OP": "?"},  # parfois il peut y avoir un tiret, parfois non
        {"LOWER": "core"},
        
        # 3) are / = / is ...
        {"IS_PUNCT": True, "OP": "*"},
        {"LOWER": {"IN": ["are", "=", "is"]}},
        {"IS_PUNCT": True, "OP": "*"},
        
        # 4) premier nombre
        {"LIKE_NUM": True, "OP": "+"},
        {"IS_PUNCT": True, "OP": "*"},
        
        # 5) "and"
        {"LOWER": "and"},
        {"IS_PUNCT": True, "OP": "*"},
        
        # 6) deuxième nombre
        {"LIKE_NUM": True, "OP": "+"},
        
        # 7) "respectively" + ponctuation potentielle
        {"LOWER": "respectively", "OP": "?"},
        {"IS_PUNCT": True, "OP": "*"}
    ]
    matcher.add("CLUST_KCORE_RESPECTIVELY", [pattern_clust_kcore])



    # --- 6) MAX K-CORE (forme simple) : "maximum k-core is X"
    pattern_kcore_1 = [
        {"LOWER": {"IN": ["maximum", "max"]}},
        {"LOWER": "k"},
        {"LOWER": "-"},
        {"LOWER": "core"},
        # ensuite "is" / "are" / ...
        {"LOWER": {"IN": ["of", "is", "=", "are"]}, "OP": "*"},
        {"LIKE_NUM": True, "OP": "+"}
    ]
    matcher.add("KCORE", [pattern_kcore_1])


    # --- 7) NUMBER OF COMMUNITIES
    # "The graph consists of X communities" / "communities equal to X"
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
    
    # Triangles description : "forming closed loops of nodes"
    pattern_triangles_desc = [
        {"LOWER": "forming"},
        {"LOWER": "closed"},
        {"LOWER": "loops"},
        {"LOWER": "of"},
        {"LOWER": "nodes"}
    ]
    matcher.add("TRIANGLES_DESC", [pattern_triangles_desc])

    # Dictionnaire résultat
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

    # On applique le matcher
    matches = matcher(doc)

    def tokens_to_float(tokens):
        """Convertit un ensemble de tokens potentiellement séparés en un float unique.
           ex: ["10", ".", "5"] -> 10.5
        """
        text_concat = "".join(t.text for t in tokens)
        try:
            return float(text_concat)
        except ValueError:
            return 0

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # ex: "NODES", "EDGES", ...
        span = doc[start:end]

        # Cas spécial : CLUST_KCORE_RESPECTIVELY => 2 nombres
        if rule_id == "CLUST_KCORE_RESPECTIVELY":
            digit_tokens = [t for t in span if t.like_num]
            if len(digit_tokens) >= 2:
                clust_val = tokens_to_float([digit_tokens[0]])
                kcore_val = tokens_to_float([digit_tokens[1]])
                info["clustering_coefficient"] = clust_val
                info["max_kcore"] = int(kcore_val) if kcore_val is not None else 0
            continue

        # Sinon, patterns classiques (1 nombre)
        digit_tokens = [t for t in span if t.like_num]
        if not digit_tokens:
            # Un match sans token numérique => ex: TRIANGLES_DESC
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

def get_sentence_features(txt):
    info = parse_graph_description(txt)
    return np.array(list(info.values()))

def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = get_sentence_features(line)
    fread.close()
    return stats

if __name__ == "__main__":
    texts = [
        # 1) Contient "The average degree is equal to 32" + "clustering... k-core are X and Y respectively"
        "This graph comprises 39 nodes and 624 edges. The average degree is equal to 32 and there are 5374 triangles in the graph. The global clustering coefficient and the graph's maximum k-core are 0.8334367245657568 and 32 respectively. The graph consists of 3 communities.",

        # 2) Contient "On average, each node is connected to 10.0..."
        "In this graph, there are 19 nodes connected by 95 edges. On average, each node is connected to 10.0 other nodes. Within the graph, there are 163 triangles. The global clustering coefficient is 0.5451505016722408. Additionally, the graph has a maximum k-core of 7 and a number of communities equal to 3.",

        # 3) Contient "On average, each node is connected to 4.6666..."
        "In this graph, there are 15 nodes connected by 35 edges. On average, each node is connected to 4.666666666666667 other nodes. Within the graph, there are 29 triangles, forming closed loops of nodes. The global clustering coefficient is 0.34523809523809523. Additionally, the graph has a maximum k-core of 3 and a number of communities equal to 2.",
    
        "In this graph, there are 49 nodes connected by 73 edges. On average, each node is connected to 2.979591836734694 other nodes. Within the graph, there are 2 triangles, forming closed loops of nodes. The global clustering coefficient is 0.03636363636363636. Additionally, the graph has a maximum k-core of 2 and a number of communities equal to 6."
    ]

    for i, txt in enumerate(texts, start=1):
        info = parse_graph_description(txt)
        print(f"=== Description {i} ===")
        print("Parsed info:", info)
        print(list(info.values()))
