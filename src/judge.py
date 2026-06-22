"""The veracity judge: build the 5W1H fact-checking prompt from a sample's clusters,
run it through the configured backend, and parse a {verdict, explanation} out of the
completion.

The prompt is the paper's, verbatim, and is identical for every backend. The transport
(openai) is injected via `set_backend()` (see src/backends.py).
"""
from __future__ import annotations

import re
import ast
import json

# Injected transport: a callable prompt:str -> completion:str. Must be set before judging.
_backend = None


def set_backend(fn):
    """Register the judge transport. `fn(prompt:str) -> str`."""
    global _backend
    _backend = fn


def _run(prompt):
    if _backend is None:
        raise RuntimeError("No judge backend set. Call src.judge.set_backend(...) first.")
    return _backend(prompt)


# --- output parsing ----------------------------------------------------------------
_VERDICT_CANON = {
    "true claim": "True Claim",
    "misleading": "Misleading",
    "not enough data": "Not enough Data",
}


def _canon_verdict(s):
    return _VERDICT_CANON.get(str(s).strip().lower())


def _first_brace_block(text):
    """Return the substring from the first '{' to its matching '}', or None."""
    if not isinstance(text, str) or "{" not in text:
        return None
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_judge_output(text):
    """Parse a judge completion into {'verdict','explanation'}, robust to varied model
    output styles, tried in order:
      1) structured dict (json.loads OR ast.literal_eval) over the cleaned text and the
         first balanced {...} block -- handles single/double quotes + prose wrap;
      2) an explicit `verdict: <label>` field anywhere in the text;
      3) the LAST occurring verdict-label phrase in free text (conclusion bias).
    Returns None if no verdict label is found.
    """
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if "```" in cleaned:   # drop code fences
        cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned).replace("```", "").strip()

    for candidate in (cleaned, _first_brace_block(text)):
        if not candidate:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
            except Exception:
                continue
            if isinstance(obj, dict) and "verdict" in obj:
                v = _canon_verdict(obj.get("verdict", ""))
                if v:
                    expl = str(obj.get("explanation", "")).strip() or text.strip()[:2000]
                    return {"verdict": v, "explanation": expl}

    m = re.search(r'verdict["\']?\s*[:=]\s*["\']?\s*(true claim|misleading|not enough data)',
                  text, re.IGNORECASE)
    if m:
        return {"verdict": _canon_verdict(m.group(1)), "explanation": text.strip()[:2000]}

    low = text.lower()
    hits = [(low.rfind(p), p) for p in _VERDICT_CANON if low.rfind(p) != -1]
    if hits:
        hits.sort()
        return {"verdict": _canon_verdict(hits[-1][1]), "explanation": text.strip()[:2000]}
    return None


# --- prompt construction -----------------------------------------------------------
_XV_SIM_THRESHOLD = 0.90   # only keep direct-search (CXV) evidence above this visual sim


def _build_narratives_str(cluster_narratives, cxv, cxt):
    filtered_cxvs = []
    for group in cxv:
        fg = []
        for item in group:
            if "summary" in item and "xv_s" in item and item["xv_s"] > _XV_SIM_THRESHOLD:
                fg.append({"summary": item["summary"],
                           "xv_s": item["xv_s"]})
        filtered_cxvs.append(fg)

    filtered_cxt = []
    for group in cxt:
        fg = []
        for item in group:
            if "summary" in item:
                fg.append({"summary": item["summary"]})
        filtered_cxt.append(fg)

    if not cluster_narratives:
        return ""
    return " ".join([
        f"Narrative {i+1} - Description: '{cluster_narratives[i] if filtered_cxvs or filtered_cxt else ''}', "
        f"CXV: [{', '.join(c['summary'] for c in filtered_cxvs[i]) if filtered_cxvs[i] else ''}], "
        f"CXT: [{{'summary': '{' '.join(c['summary'] for c in filtered_cxt[i])}'}}];"
        for i in range(min(len(cluster_narratives), len(filtered_cxvs), len(filtered_cxt)))
    ])


def _build_prompt(t_claim, clustered_narratives_str):
    return f"""You are a helpful Fact-Checking Assistant. You follow ALL Instructions strictly. Your task is to evaluate whether the Text Claim ('t_claim') and the Claim Image are used in context using evidence grouped into Narratives.

Each Narrative includes:
(1) 'Clustered Narratives', a description of the i-th narrative;
(2) 'CXV', direct evidence derived from 't_claim'; and
(3) 'CXT', reverse evidence obtained by reverse searching with the Claim Image.
Reverse search evidence ('CXT') is the strongest indicator of the Claim Image provenance.

### 1. Text Alignment Rules
Your task is to determine whether the 'summary' *supports*, *conflicts* the 't_claim' based on the following categories: **Location, Named Person, Date, Main Topic, and Common Objects**.
Perform **intelligent and contextual named entity extraction and comparison** to find support or conflict.

Follow these strict rules for comparison:

#### (a) *supports*
- **Location**: If and only if Location entity is present in both 'summary' and 't_claim', they must refer to the same place geographically. (e.g., "France" and "Paris"; "Empire State Building" and "New York").
- **Named Person**: If and only if Named Person entity appears in both 'summary' and 't_claim', they must refer to the same individual. (e.g., "Obama" and "Barack Obama").
- **Date**: If and only if Date entity (Month and/or Year) appears in both 'summary' and 't_claim', they must refer to the same time frame. (e.g., "March 2020" and "2020"; "3rd March" and "March").
- **Main Topic**: If and only if the main topic is clearly identifiable in both 'summary' and 't_claim', they must refer to similar themes. (e.g., "Protest for Human Rights" and "Demonstrations for Human Rights"; "Flood" and "Heavy Rain").
- **Common Objects**: If and only if some common named entities (objects, places, or people) appear in both 'summary' and 't_claim'.

#### (b) *conflicts*
- **Location**: If and only if  Location entity is present in both 'summary' and 't_claim' and they refer to different places. (e.g., "India" and "Paris"; "Empire State Building" and "London").
- **Named Person**: If and only if Named Person entity appears in both 'summary' and 't_claim' and refers to different individuals. (e.g., "Obama" and "Trump").
- **Date**: If and only if Date entity appears in both 'summary' and 't_claim' and they refer to different time frames. (e.g., "March 2020" and "2023"; "3rd March" and "June").
- **Main Topic**: If and only if the main topic in 'summary' and 't_claim' refers to different subjects.
- **Common Objects**: If and only if no common named entities exist between 'summary' and 't_claim'.

#### (c) Decision Rules
1. Ensure named entity extraction and comparison are performed as per the above rules.
2. If there is a conflict  in any of **Location, Named Person, or Date**, classify as *conflicts*.
3. If there is no conflict in any of **Location, Named Person, or Date**, then:
   - If there is support   in  any of **Location, Named Person, or Date**, classify as *supports*.
   - If there is no support in any of **Location, Named Person, or Date**, but support in any of **Main Topic or Common Objects**, classify as *supports*.
   - If there is no support in any of **Location, Named Person, or Date**, and any of **Main Topic or Common Objects** conflict, classify as *conflicts*.


5. Understand that if the evidence has named entites present then it must be classified as *supports* or *conflicts*
   If the entities present in the evidence are unrelated to the claim then this is *conflict* . Such evidences are very important and relevant in fact checking as as it implies out-of-context usage.
   Thus focus on entity extraction carefuly.
   The only case when an evidence is irrelvant is when it does not have any named entites.


#### Follow these **Instructions** to Evaluate ALL CXT , CXV For each Narratives.
Remember a narrative can have Multiple summary elements in both its CXT and CXV, You have to process all summaries present in both  CXT and CXV:

### 2. Evaluating Reverse Search Evidence ('CXT')
Reverse search evidence determines the provenance of the Claim Image and is the Most Important piece of Evidence. If 'CXT' is non-empty:
- Compare each item's 'summary' with 't_claim' in term of **Location**, **Named Person** ,**Date** , **Main Topic**, **Common Objects** strictly using **Text Alignment Rules**.
- Classification using **Text Alignment Rules**:
   - If 'summary' *supports* 't_claim', this is *Relevant Reverse Search Evidence* and evidence for a **True Claim**.
   - If 'summary' *conflicts* with 't_claim', this is *Relevant Reverse Search Evidence* and evidence for **Misleading**.
   - If 'summary' is empty or has no named entities this can be ignored
   - If 'CXT' is an empty list, then it is *Not Relevant Reverse Search Evidence*.

### 3. Evaluating Direct Search Evidence ('CXV')
Direct search evidence ('CXV') is derived using 't_claim'. If 'CXV' is non-empty:
- Check **visual similarity score ('xv_s')**.
- Compare each item's 'summary' with 't_claim' in term of **Location**, **Named Person** ,**Date** , **Main Topic**, **Common Objects** strictly using **Text Alignment Rules**.
- Classification using **Text Alignment Rules**:
  - If visual similarity is **high**:
    - If 'summary' *supports* 't_claim', classify as **True Claim**.
    - If 'summary' *conflicts* with 't_claim', classify as **Misleading**.
    - If 'summary' is empty or has no named entities this can be ignored
  - If visual similarity is **low**, ignore the evidence.

### 4. Priortise Narratives for Final Judgment. This is the most important RULE and must be strictly followed to every word.
- Ignore Narratives where both 'CXT' and 'CXV' are empty.
- Understand that CXT is the most relevant evidence, thus prioritise narratives with CXT (ONLY CXT OR WITH CXT AND CXV)  over Narratives with only CXV for your judgement
- If Narratives with *Relevant Reverse Search Evidence (CXT)* exist, then you MUST use ONLY THESE for Final Decision, AND ignore the rest of Narratives:
  - If any Narrative with 'CXT' *supports*, classify as **True Claim**.
  - If no Narrative with 'CXT' *supports*, and all Narratives with 'CXT' *conflict* or *ignore*, or is  *unrelated* classify as **Misleading**.
- If there are no Narratives with *Relevant Reverse Search Evidence (CXT)*, ONLY THEN can you use Narratives with ONLY *Relevant Direct Search Evidence (CXV)*:
  - If any Narrative with 'CXV' *supports*, classify as **True Claim**.
  - If no Narrative with 'CXV' *supports*, and all Narratives with 'CXV' *conflict* or *ignore*, classify as **Misleading**.

### 5. Final Judgment
- Output **'verdict'** as one of ['Misleading', 'True Claim'].
- Prefer ['Misleading', 'True Claim'] if *Relevant Evidence* exists.
- MOST IMPORTANT RULE: Remember for your final judgement,  YOU CAN USE Narratives with CXT absent/ empty , ONLY IF THERE ARE NO NARRATIVES exist WITH non empty CXT. You cant ignore this rule.


### Inputs:
- **Text Claim**: '{t_claim}'
- **Clustered Narratives**: '{clustered_narratives_str}'

### Expected Output:
Return a **valid Python dictionary** formatted for `ast.literal_eval()` parsing, with:
1) **'verdict'**: One of ['Misleading', 'True Claim'].
2) **'explanation'**: A summary of reasoning, explicitly mentioning what was supported, what was not, and what conflicted in terms of **Location, Named Person, Date, Main Topic, Common Objects** while following **Text Alignment Rules**.

Return only the dictionary without markdown, extra text, or code block markers."""


def judge_narratives(clusters, t_claim):
    """Judge a sample's clusters. Returns (verdict, explanation). On a parse failure after
    retries returns ('judge_narratives fail', 'parse_failed:: ...')."""
    cluster_narratives, cxv, cxt = [], [], []
    for cluster in clusters:
        cluster_narratives.append(cluster.get("common_narrative", ""))
        cxv.append(cluster.get("xv", []))
        cxt.append(cluster.get("xt", []))

    narratives_str = _build_narratives_str(cluster_narratives, cxv, cxt)
    prompt = _build_prompt(t_claim, narratives_str)

    result, last_raw = None, ""
    for _ in range(2):                  # one retry on an unparseable completion
        try:
            last_raw = _run(prompt)
            result = _parse_judge_output(last_raw)
            if result is not None and result.get("verdict"):
                break
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Judge error: {e}. Retrying...")

    if result is None or not result.get("verdict"):
        return "judge_narratives fail", f"parse_failed:: {str(last_raw)[:1500]}"
    return result["verdict"], result["explanation"]
