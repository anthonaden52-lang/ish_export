"""Turn one token's indicator row into a verdict: BUY CANDIDATE / WATCH / AVOID with a 0-100 score.

Two score sources, ALWAYS labelled:
  MODEL       the trained gradient-boosting model. If a separate temporal calibrator was saved,
              score = calibrated probability x 100; otherwise it uses the raw model output and says
              "uncalibrated". Neither value is a guarantee, and it is never labelled "confidence".
  ANALYTICAL  hand-written rules, used only when no trained model exists. Not validated, not a
              probability, and no confidence is shown for it.
Run evaluate.py to see whether either score has actually predicted anything.
"""
import os

import numpy as np

from model_store import predict_probability

BUY_SCORE = float(os.getenv("BUY_SCORE", "80"))
WATCH_SCORE = float(os.getenv("WATCH_SCORE", "60"))


def _ok(x):
    return x is not None and x == x and not np.isinf(x)


def _ramp(x, lo, hi):
    """0 at/below lo, 1 at/above hi, linear between. NaN -> 0 (no evidence = no credit)."""
    if not _ok(x):
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def analytical_score(r):
    """Returns (score 0-100, list of (points, max, text))."""
    parts = []

    def add(name, frac, mx, good, bad):
        pts = frac * mx
        parts.append((pts, mx, good if frac >= 0.6 else bad if frac <= 0.3 else None))

    mom = np.mean([_ramp(r.get("return_5"), 0, 0.05), _ramp(r.get("return_15"), 0, 0.10),
                   1.0 if _ok(r.get("macd_spread")) and r["macd_spread"] > 0 else 0.0,
                   1.0 if _ok(r.get("ema_spread")) and r["ema_spread"] > 0 else 0.0])
    add("momentum", mom, 25, "positive price momentum", "weak or negative momentum")
    add("volume", _ramp(r.get("volume_accel"), 1.0, 3.0), 20,
        "strong volume acceleration", "no volume acceleration")
    bp = np.mean([_ramp(r.get("buy_sell_ratio"), 1.0, 2.0), _ramp(r.get("buy_sell_ratio_1h"), 1.0, 1.5)])
    add("pressure", bp, 15, "buyers outnumber sellers", "sell pressure")
    liq = _ramp(r.get("liquidity_usd"), 25_000, 250_000) * 0.6 + \
        (1.0 - _ramp(-(r.get("liquidity_change_15") or 0), 0.0, 0.2)) * 0.4
    add("liquidity", liq, 20, "healthy liquidity", "thin or falling liquidity")
    rsi = r.get("rsi14")
    rsi_f = 0.0 if not _ok(rsi) else (1.0 if 45 <= rsi <= 70 else 0.5 if 70 < rsi <= 80 else 0.2)
    add("rsi", rsi_f, 10, "RSI in a healthy range", "RSI overbought or very weak")
    add("age", _ramp(r.get("pair_age_minutes"), 30, 360), 10, "pair has some track record", "very new pair")
    total = sum(p for p, _, _ in parts)
    return round(total), [t for _, _, t in parts if t]


def risk_level(r, safety):
    """Low / Medium / High / Extreme from facts, not from the score."""
    if safety and safety.get("passed") is False:
        return "Extreme"
    pts = 0
    if _ok(r.get("liquidity_usd")) and r["liquidity_usd"] < 60_000:
        pts += 1
    if _ok(r.get("pair_age_minutes")) and r["pair_age_minutes"] < 180:
        pts += 1
    if _ok(r.get("volatility_15")) and r["volatility_15"] > 0.05:
        pts += 1
    if _ok(r.get("liquidity_change_15")) and r["liquidity_change_15"] < -0.05:
        pts += 1
    if safety:
        if safety.get("passed") is None:
            pts += 2  # not verified
        t10 = (safety.get("metrics") or {}).get("top10_pct")
        if t10 is not None and t10 > 35:
            pts += 1
        pts += len(safety.get("warn") or [])
    return "Low" if pts == 0 else "Medium" if pts <= 2 else "High" if pts <= 4 else "Extreme"


def score_row(r, bundle):
    """Score only (no safety yet). Returns dict(score, source, prob, reasons)."""
    if bundle is not None:
        feats = bundle["features"]
        x = np.array([[r.get(f, np.nan) for f in feats]], dtype=float)
        prob = float(predict_probability(bundle, x)[0])
        _, reasons = analytical_score(r)  # reasons only: plain-language facts about the row
        return {"score": round(prob * 100), "source": "MODEL", "prob": prob,
                "threshold": bundle["threshold"], "calibration_method": bundle.get("calibration_method", "uncalibrated"),
                "reasons": reasons}
    s, reasons = analytical_score(r)
    return {"score": s, "source": "ANALYTICAL", "prob": None, "threshold": None, "reasons": reasons}


def decide(r, bundle, safety_fn):
    """Full verdict. safety_fn(row) -> safety dict; only called for rows that would otherwise pass."""
    sc = score_row(r, bundle)
    score, src = sc["score"], sc["source"]
    meets_buy = score >= BUY_SCORE and (src != "MODEL" or sc["prob"] >= sc["threshold"])
    verdict = "BUY CANDIDATE" if meets_buy else "WATCH" if score >= WATCH_SCORE else "AVOID"
    safety = None
    if verdict != "AVOID":
        safety = safety_fn(r)
        if safety["passed"] is False:
            verdict = "AVOID"
        elif safety["passed"] is None and verdict == "BUY CANDIDATE":
            verdict = "WATCH"  # cannot buy what has not been verified
    else:
        # still apply the free row-level checks so obvious rugs say why
        from safety import row_checks
        hard, warn = row_checks(r)
        safety = {"passed": False if hard else None, "hard": hard, "warn": warn, "metrics": {}, "not_checked": []}
        if hard:
            verdict = "AVOID"
    risk = risk_level(r, safety)
    if verdict == "BUY CANDIDATE" and risk in ("High", "Extreme"):
        verdict = "WATCH"
    reasons = list(sc["reasons"])
    reasons += [f"SAFETY: {h}" for h in safety.get("hard", [])]
    reasons += [f"note: {w}" for w in safety.get("warn", [])]
    return {"verdict": verdict, "score": score, "source": src, "prob": sc["prob"],
            "risk": risk, "reasons": reasons, "safety": safety,
            "calibration_method": sc.get("calibration_method", "uncalibrated")}


ICON = {"BUY CANDIDATE": "🟢", "WATCH": "🟡", "AVOID": "🔴"}


def format_card(r, d):
    lines = [f"{ICON[d['verdict']]} {d['verdict']}", "",
             f"Token: {r.get('symbol', '?')}  ({str(r['token_address'])[:6]}...)"]
    if d["source"] == "MODEL":
        lines.append(f"AI Score: {d['score']}/100  (trained model)")
        if d.get("calibration_method") == "temporal_platt_logistic":
            lines.append(f"Model probability (temporally calibrated, not a guarantee): {d['prob']:.0%}")
        else:
            lines.append(f"Model probability (uncalibrated): {d['prob']:.0%}")
    else:
        lines.append(f"Analytical Score: {d['score']}/100  (rule-based, NO trained model yet)")
        lines.append("Confidence: n/a")
    lines.append(f"Risk: {d['risk']}")
    lines.append("Reason: " + ("; ".join(d["reasons"]) if d["reasons"] else "no standout factors"))
    return "\n".join(lines)
