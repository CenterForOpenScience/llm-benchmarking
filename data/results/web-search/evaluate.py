#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit


# URL utilities (from your eval script)
_TRAILING_PUNCT = " \t\r\n).,;]}>\"'"

def _extract_json(text: str) -> Dict[str, Any]:
    if not text: 
        return {}
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return {}

def _get_predicted_urls(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = pred.get("result")
    
    # If result is None/Empty, try parsing 'raw_response'
    if not result and "raw_response" in pred:
        result = _extract_json(str(pred["raw_response"]))

    result = result or {}
    urls = result.get("urls") or []
    
    if not isinstance(urls, list):
        return []
        
    out: List[Dict[str, Any]] = []
    for u in urls:
        if isinstance(u, dict) and "url" in u:
            out.append(u)
        elif isinstance(u, str):
            out.append({"url": u})
    return out

def _sanitize_url(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.strip(_TRAILING_PUNCT)
    if s.startswith("www."):
        s = "https://" + s
    return s

def _looks_like_url(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("www.")

def _normalize_url(url: str) -> Tuple[str, str]:
    """
    Returns (strict, loose):
      strict: netloc + path + (?query if present)
      loose : netloc + path
    """
    u = _sanitize_url(url)
    if u is None:
        return ("", "")

    parts = urlsplit(u)
    if not parts.netloc and parts.path and not parts.scheme:
        parts = urlsplit("https://" + u)

    netloc = (parts.netloc or "").lower()
    path = (parts.path or "").lower()
    query = parts.query or ""

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    base = netloc + path
    strict = base + (("?" + query) if query else "")
    loose = base
    return strict, loose

# Data structures
@dataclass(frozen=True)
class GoldResource:
    rid: str
    kind: str
    required: bool
    filename: str
    rl: str
    aliases: List[str]

    def acceptable_url_keys(self) -> Set[str]:
        keys: Set[str] = set()
        for raw in [self.rl] + [a for a in (self.aliases or []) if _looks_like_url(str(a))]:
            s = _sanitize_url(raw)
            if not s:
                continue
            strict, loose = _normalize_url(s)
            if strict:
                keys.add(strict)
            if loose:
                keys.add(loose)
        return keys

@dataclass
class CaseMetrics:
    case_id: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    hit_any: int
    hit_all: int


# Matching + evaluation
def _get_predicted_urls(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = pred.get("result") or {}
    urls = result.get("urls") or []
    if not isinstance(urls, list):
        return []
    out: List[Dict[str, Any]] = []
    for u in urls:
        if isinstance(u, dict) and "url" in u:
            out.append(u)
        elif isinstance(u, str):
            out.append({"url": u})
    return out

def _choose_best_candidate(
    candidates: List[GoldResource],
    pred_obj: Dict[str, Any],
    matched: Set[str],
) -> Optional[GoldResource]:
    if not candidates:
        return None

    unmatched = [c for c in candidates if c.rid not in matched]
    if unmatched:
        candidates = unmatched

    pred_kind = (pred_obj.get("kind") or "").strip().lower()
    pred_text = " ".join(str(pred_obj.get(k) or "") for k in ("resource_name", "why_needed", "url")).lower()

    def score(gr: GoldResource) -> float:
        s = 0.0
        if gr.required:
            s += 0.2
        if pred_kind and gr.kind and pred_kind == gr.kind.lower():
            s += 0.2
        fn = (gr.filename or "").lower()
        rid = (gr.rid or "").lower()
        if fn and fn in pred_text:
            s += 1.0
        if rid and rid in pred_text:
            s += 0.5
        s -= 0.01 * len(gr.acceptable_url_keys())
        return s

    return max(candidates, key=score)

def evaluate_one_case(gold_case: Dict[str, Any], pred_urls: List[Dict[str, Any]]) -> CaseMetrics:
    case_id = int(gold_case["id"])
    gold_resources_raw = gold_case.get("gold_resources") or []

    gold_resources: List[GoldResource] = []
    for r in gold_resources_raw:
        rl = r.get("rl") or r.get("url") or ""
        gold_resources.append(
            GoldResource(
                rid=str(r.get("id")),
                kind=str(r.get("kind") or "").strip(),
                required=bool(r.get("required", True)),
                filename=str(r.get("filename") or ""),
                rl=str(rl),
                aliases=list(r.get("aliases") or []),
            )
        )

    required_ids = {gr.rid for gr in gold_resources if gr.required}

    urlkey_to_candidates: Dict[str, List[GoldResource]] = {}
    for gr in gold_resources:
        for k in gr.acceptable_url_keys():
            urlkey_to_candidates.setdefault(k, []).append(gr)

    seen_pred_keys: Set[str] = set()
    unique_pred: List[Dict[str, Any]] = []
    for obj in pred_urls:
        raw_url = _sanitize_url(obj.get("url"))
        if not raw_url:
            continue
        strict, loose = _normalize_url(raw_url)
        key = strict or loose
        if not key or key in seen_pred_keys:
            continue
        seen_pred_keys.add(key)
        unique_pred.append({**obj, "_norm_strict": strict, "_norm_loose": loose})

    matched_required: Set[str] = set()
    matched_any_gold: Set[str] = set()
    spurious_pred_urls: Set[str] = set()

    for obj in unique_pred:
        strict = obj.get("_norm_strict", "")
        loose = obj.get("_norm_loose", "")
        candidates: List[GoldResource] = []
        
        # Try Exact Matches first
        if strict and strict in urlkey_to_candidates:
            candidates.extend(urlkey_to_candidates[strict])
        if loose and loose in urlkey_to_candidates:
            candidates.extend(urlkey_to_candidates[loose])

        # If no exact match, try Relaxed/Prefix Matches
        if not candidates:
            for gold_key, gold_cands in urlkey_to_candidates.items():
                # Check if Prediction is a sub-page of Gold Alias
                # e.g. Gold: "oecd.org", Pred: "oecd.org/data/report.pdf"
                if strict.startswith(gold_key) or loose.startswith(gold_key):
                    candidates.extend(gold_cands)
                
                # Check if Gold is a sub-page of Prediction (optional, covers generic landing pages)
                # e.g. Gold: "oecd.org/report.pdf", Pred: "oecd.org"
                elif gold_key.startswith(strict) or gold_key.startswith(loose):
                    candidates.extend(gold_cands)

        # dedupe candidates by rid
        seen = set()
        deduped: List[GoldResource] = []
        for c in candidates:
            if c.rid in seen:
                continue
            seen.add(c.rid)
            deduped.append(c)
        candidates = deduped

        if not candidates:
            spurious_pred_urls.add(strict or loose)
            continue

        chosen = _choose_best_candidate(candidates, obj, matched_any_gold)
        if chosen is None:
            continue

        matched_any_gold.add(chosen.rid)
        if chosen.required:
            matched_required.add(chosen.rid)

    tp = len(matched_required)
    fn = len(required_ids - matched_required)
    fp = len(spurious_pred_urls)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    hit_any = 1 if tp > 0 else 0
    hit_all = 1 if fn == 0 else 0

    return CaseMetrics(
        case_id=case_id,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        hit_any=hit_any,
        hit_all=hit_all,
    )

def _aggregate_case_metrics(case_metrics: List[CaseMetrics]) -> Dict[str, float]:
    if not case_metrics:
        return {
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "hit_any": 0.0, "hit_all": 0.0,
            "n_cases": 0.0,
        }

    TP = sum(m.tp for m in case_metrics)
    FP = sum(m.fp for m in case_metrics)
    FN = sum(m.fn for m in case_metrics)
    micro_p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    micro_r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    macro_p = sum(m.precision for m in case_metrics) / len(case_metrics)
    macro_r = sum(m.recall for m in case_metrics) / len(case_metrics)
    macro_f = sum(m.f1 for m in case_metrics) / len(case_metrics)

    hit_any = sum(m.hit_any for m in case_metrics) / len(case_metrics)
    hit_all = sum(m.hit_all for m in case_metrics) / len(case_metrics)

    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "hit_any": hit_any,
        "hit_all": hit_all,
        "n_cases": float(len(case_metrics)),
    }

def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, math.sqrt(var)

def evaluate_models(groundtruth: List[Dict[str, Any]], predictions: List[Dict[str, Any]], model_key: str) -> Dict[str, Any]:
    gt_by_id = {int(c["id"]): c for c in groundtruth}

    grouped: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for pred in predictions:
        model = str(pred.get(model_key) or pred.get("model") or "unknown")
        cid = pred.get("id") or pred.get("case_id") or pred.get("study_id")
        if cid is None:
            continue
        try:
            cid = int(cid)
        except Exception:
            continue
        grouped.setdefault(model, {}).setdefault(cid, []).append(pred)

    results: Dict[str, Any] = {}

    for model, case_runs in grouped.items():
        max_runs = max((len(v) for v in case_runs.values()), default=0)
        per_run: List[Dict[str, float]] = []

        for run_idx in range(max_runs):
            case_metrics: List[CaseMetrics] = []
            for cid, runs in case_runs.items():
                if run_idx >= len(runs):
                    continue
                if cid not in gt_by_id:
                    continue
                pred = runs[run_idx]
                pred_urls = _get_predicted_urls(pred)
                case_metrics.append(evaluate_one_case(gt_by_id[cid], pred_urls))
            per_run.append(_aggregate_case_metrics(case_metrics))

        summary: Dict[str, Any] = {"n_runs": len(per_run)}
        for k in [
            "micro_precision", "micro_recall", "micro_f1",
            "macro_precision", "macro_recall", "macro_f1",
            "hit_any", "hit_all",
        ]:
            vals = [r.get(k, 0.0) for r in per_run]
            m, s = _mean_std(vals)
            summary[k] = m
            summary[k + "_std"] = s

        results[model] = {"per_run": per_run, "summary": summary}

    return results
def load_prediction_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.lstrip("\ufeff").replace("\x00", "")
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    out: List[Dict[str, Any]] = []
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError as e:
            ctx = text[max(0, e.pos - 120):min(n, e.pos + 120)]
            raise SystemExit(f"Could not parse predictions in {path} at line {e.lineno} col {e.colno}: {e.msg}\nContext: {ctx!r}")
        i = j
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] == ",":
            i += 1
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    out.append(item)
        elif isinstance(obj, dict):
            out.append(obj)
    return out

def load_predictions(paths: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in paths:
        if p.is_dir():
            files = sorted([*p.glob("*.json"), *p.glob("*.jsonl")])
            records.extend(load_predictions(files))
        else:
            records.extend(load_prediction_file(p))
    return records

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold_set.json", help="Path to gold_set.json")
    ap.add_argument("--predictions",nargs="+",help="Path to an existing JSON or JSONL file containing predictions")
    ap.add_argument("--model-key", default="search_model", help="Field name to store model id")
    ap.add_argument("--metrics-out", default="", help="If set, write metrics JSON to this path (overall, all models)")
    args = ap.parse_args()

    gold_path = Path(args.gold).resolve()
    groundtruth = load_json(gold_path)
    if not isinstance(groundtruth, list):
        raise SystemExit(f"Gold set must be a JSON list of cases. Got: {type(groundtruth)}")
    
    pred_paths = [Path(p).resolve() for p in args.predictions]
    all_predictions = load_predictions(pred_paths)

    # Evaluate all loaded predictions
    metrics = evaluate_models(groundtruth, all_predictions, model_key=args.model_key)

    # Print compact summaries
    for model, blob in metrics.items():
        s = blob["summary"]
        print(f"\n== {model} ==")
        print(f"Macro P/R/F1: {s['macro_precision']:.4f}/{s['macro_recall']:.4f}/{s['macro_f1']:.4f} (std {s['macro_f1_std']:.4f})")
        print(f"Micro P/R/F1: {s['micro_precision']:.4f}/{s['micro_recall']:.4f}/{s['micro_f1']:.4f} (std {s['micro_f1_std']:.4f})")
        print(f"hit@any: {s['hit_any']:.4f} (std {s['hit_any_std']:.4f})")
        print(f"hit@all: {s['hit_all']:.4f} (std {s['hit_all_std']:.4f})")
        print(f"n_runs: {s['n_runs']}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out).resolve()
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[OK] Wrote metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

