"""Evidence clustering: KMeans (exact k) and Dynamic/Agglomerative (similarity threshold).

Self-contained: text is embedded with a locally-loaded SentenceTransformer (no service,
no image model). Both functions take the flat evidence pools `xv` (direct/visual search)
and `xt` (reverse-image search) and return a list of cluster dicts
`{common_narrative, xv, xt}` -- the schema the judge consumes.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config.clustering import EMBED_MODEL, cluster_params


def get_text_embedding(text):
    """Embed text with a locally-loaded SentenceTransformer (loaded once, 384-d)."""
    if not hasattr(get_text_embedding, "_model"):
        print(f"[clustering] loading SentenceTransformer {EMBED_MODEL} ...")
        get_text_embedding._model = SentenceTransformer(EMBED_MODEL)
    return get_text_embedding._model.encode(text, convert_to_numpy=True)


def _embed_items(items, claim_text=None):
    """Embed evidence items from their `summary`. If claim_text is given, make the
    embedding claim-aware by prepending it (used by the Dynamic scheme)."""
    vecs = []
    for item in items:
        text = (item.get("summary", "") or "").strip()
        if claim_text:
            text = f"{claim_text.strip()} [SEP] {text}"
        vecs.append(get_text_embedding(text or "empty document"))
    return np.array(vecs) if vecs else np.array([])


def _representative_narrative(item):
    txt = (item.get("summary", "") or "").strip()
    return txt or "Single evidence item."


def generate_clusters_kmeans(xv, xt, claim_text, num_clusters):
    """Partition xv+xt into exactly `num_clusters` KMeans clusters."""
    all_evidence = xv + xt
    if not all_evidence:
        return [{"common_narrative": "", "xv": [], "xt": []} for _ in range(num_clusters)]

    X = _embed_items(all_evidence)               # KMeans embeds evidence text only
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    clusters = [{"common_narrative": "", "xv": [], "xt": []} for _ in range(num_clusters)]
    for idx, evidence in enumerate(all_evidence):
        bucket = "xv" if idx < len(xv) else "xt"
        clusters[labels[idx]][bucket].append(evidence)

    for ci in range(num_clusters):
        members = [i for i, lbl in enumerate(labels) if lbl == ci]
        if members:
            dists = np.linalg.norm(X[members] - centroids[ci], axis=1)
            closest = all_evidence[members[int(np.argmin(dists))]]
            clusters[ci]["common_narrative"] = _representative_narrative(closest)
    return clusters


def generate_clusters_dynamic(xv, xt, claim_text, cap, similarity_threshold):
    """Agglomerative clustering at a cosine-similarity threshold; count capped at `cap`."""
    all_evidence = xv + xt
    if cap <= 0 or not all_evidence:
        return []

    X = _embed_items(all_evidence, claim_text=claim_text)   # Dynamic is claim-aware
    if X.size == 0:
        return []

    if X.shape[0] < 2:   # one item -> one cluster
        return [{
            "common_narrative": _representative_narrative(all_evidence[0]),
            "xv": list(xv), "xt": list(xt),
        }]

    distance_threshold = max(0.0, min(1.0, 1 - similarity_threshold))
    agg = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold,
    )
    try:
        labels = agg.fit_predict(X)
    except ValueError:
        labels = np.zeros(X.shape[0], dtype=int)

    # Group items by emergent label.
    groups = {}
    for idx, evidence in enumerate(all_evidence):
        g = groups.setdefault(labels[idx], {"idx": [], "emb": [], "xv": [], "xt": []})
        g["idx"].append(idx)
        g["emb"].append(X[idx])
        (g["xv"] if idx < len(xv) else g["xt"]).append(evidence)

    clusters = []
    for label in sorted(groups.keys())[:cap]:    # cap the number of clusters
        g = groups[label]
        emb = np.array(g["emb"])
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        centroid = emb.mean(axis=0).reshape(1, -1)
        closest_local = int(np.argmax(cosine_similarity(emb, centroid).flatten()))
        rep = all_evidence[g["idx"][closest_local]]
        clusters.append({
            "common_narrative": _representative_narrative(rep),
            "xv": g["xv"], "xt": g["xt"],
        })
    return clusters


def cluster_sample(sample, algo, tag):
    """Re-cluster one sample's flat evidence under a config. Returns the cluster list."""
    claim = sample.get("Claim", "")
    xv = sample.get("xv", []) or []
    xt = sample.get("xt", []) or []
    n_items = len(xv) + len(xt)
    num_or_cap, threshold = cluster_params(algo, tag)
    if algo == "K":
        eff_k = min(num_or_cap, n_items)      # cannot make more clusters than points
        if eff_k < 1:
            return []
        return generate_clusters_kmeans(xv, xt, claim, eff_k)
    return generate_clusters_dynamic(xv, xt, claim, num_or_cap, threshold)
