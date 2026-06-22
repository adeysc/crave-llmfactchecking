"""JSON load/save + the filename conventions for data and outputs.

data/    {ds}_K4_wo.json   canonical baseevd (raw wo_refine evidence + wo verdicts)
         {ds}_K4_wr.json   with_refine + stored GPT-4o verdicts (paper reference)
outputs/ {ds}_{cfg}_wo.json                re-clusterings        (generate_clusters)
         {ds}_{cfg}_judged_{judge}.json    judged configs        (judge_clusters)
         experiment_{ds}_{judge}.json      per-dataset summary    (judge_clusters)
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

import numpy as np

from config.paths import DATA, OUTPUTS


def _np_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# MINIMAL evidence schema (repo-wide, parity with tcss_ad_gather_score):
#   xt -> {summary, domain}      xv -> {summary, domain, xv_s}
# caption/title/image_path/date/page_link are all dropped. `domain` is derived from `page_link`
# when not already stored (older _wr baseevd kept page_link but not domain).
_KEEP_XT = ("summary", "domain")
_KEEP_XV = ("summary", "domain", "xv_s")


def _domain_of(item):
    dom = item.get("domain")
    if dom not in (None, ""):
        return dom
    pl = item.get("page_link", "") or ""
    return urlparse(pl).netloc if pl else ""


def _slim_item(item, keep):
    if not isinstance(item, dict):
        return item
    out = {}
    if "summary" in keep:
        out["summary"] = item.get("summary", "")
    if "domain" in keep:
        out["domain"] = _domain_of(item)
    if "xv_s" in keep and "xv_s" in item:
        out["xv_s"] = item["xv_s"]
    return out


def _slim_evidence(obj):
    """Recursively rebuild a structure, reducing every `xt`/`xv` list to the minimal schema above
    (non-mutating). Other keys (Claim, clusters, judgement_*, common_narrative, ...) pass through."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "xt" and isinstance(v, list):
                out[k] = [_slim_item(it, _KEEP_XT) for it in v]
            elif k == "xv" and isinstance(v, list):
                out[k] = [_slim_item(it, _KEEP_XV) for it in v]
            else:
                out[k] = _slim_evidence(v)
        return out
    if isinstance(obj, list):
        return [_slim_evidence(x) for x in obj]
    return obj


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_slim_evidence(data), f, indent=2, ensure_ascii=False, default=_np_default)


# --- path helpers -----------------------------------------------------------------
def data_path(dataset, prov):
    """prov in {'wo','wr'} -> data/{dataset}_K4_{prov}.json"""
    return DATA / f"{dataset}_K4_{prov}.json"


def cluster_path(dataset, cfg):
    """cfg like 'K2','D75' -> outputs/{dataset}_{cfg}_wo.json"""
    return OUTPUTS / f"{dataset}_{cfg}_wo.json"


def config_input_path(dataset, cfg):
    """Where judge_clusters reads a config's clusters from. K4 == the baseevd in data/."""
    if cfg == "K4":
        return data_path(dataset, "wo")
    return cluster_path(dataset, cfg)


def judged_path(dataset, cfg, judge):
    return OUTPUTS / f"{dataset}_{cfg}_judged_{judge}.json"


def summary_path(dataset, judge):
    return OUTPUTS / f"experiment_{dataset}_{judge}.json"
