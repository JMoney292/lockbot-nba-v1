import math
from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

"""
NBA Predictor  (v1.2)
- Market-led, tiny power/form nudges.
- Convert spread -> win% with calibrated curve (NBA: -5.5 ≈ ~70% ML).
- Split scores directly from (total, spread).
- FIX: Side decision now uses RAW model edge vs the book (not tiny blended move),
      so you'll get dog +book when our model is ≥ threshold the other way.
- Slightly looser anchor so real edges can trigger (still conservative).
"""

# =================== Tunables (small & stable) ===================
SHOW_DEBUG = True  # turn on so you can see edges that drive the pick

# Calibrated spread -> win% (NBA)
K_SPREAD = 0.15             # => -5.5 ~70% ML
PROB_SQUASH_TEMP = 0.90     # compress probabilities toward 0.5

# Market anchoring (slightly looser than before so an edge can fire)
WMKT_SPREAD = 0.75          # 75% market / 25% model for spread (was 0.85)
WMKT_TOTAL  = 0.90          # totals still very market-led

# Tiny model components
HFA_PTS = 1.8               # home-court baseline (pts)
BETA_POWER = 0.06           # pts per (home_power - away_power)
BETA_FORM  = 0.04           # pts per (homeL5 - awayL5)

# Caps (keep model in its lane)
SPREAD_NUDGE_CAP = 3.0      # max our model can differ from book (raw) ±3.0
TOTAL_NUDGE_CAP  = 2.0      # max our model can push total (raw) ±2.0
FINAL_TOTAL_CLAMP = 4.0     # final total stays within ±4 of market

# Recommendation thresholds — RAW edge vs book (like our NFL)
EDGE_TO_LAY_FAV  = 2.0      # lay favorite if model is ≥ 2.0 pts more pro-fav than book
EDGE_TO_TAKE_DOG = 2.0      # take dog +book if model is ≥ 2.0 pts more pro-dog than book
CONF_BASEMAX = 78.0         # cap base confidence

st.set_page_config(page_title="NBA — NFL-Style Minimal", layout="wide")
st.title("🏀 NBA Predictor — NFL-Style Minimal")
st.caption("Market-led picks with tiny power/form nudges. Scores split from (total, spread). No pace, no extras.")

# =================== Helpers ===================
def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class Inputs:
    home_team: str
    away_team: str
    home_power: float    # ELO/power (relative diffs matter)
    away_power: float
    market_spread_home_minus: float  # home favored = negative (e.g., -5.5)
    market_total: float
    home_l5: float = 0.0             # optional recent form (avg margin)
    away_l5: float = 0.0

def predict(si: Inputs) -> Dict:
    # -------- Model margin (home perspective; + = home by X) --------
    power_diff = si.home_power - si.away_power
    form_diff  = si.home_l5 - si.away_l5

    model_margin = HFA_PTS + (BETA_POWER * power_diff) + (BETA_FORM * form_diff)
    model_margin = clamp(model_margin, -12, 12)  # safety cap

    # Convert to home-spread sign (negative = home favored)
    model_home_spread = -model_margin

    # -------- RAW edge vs the book --------
    # Positive raw_edge_dog  -> our model is LESS pro-home than the book (toward dog)
    # Positive raw_edge_fav  -> our model is MORE pro-home than the book (toward favorite)
    raw_diff = model_home_spread - si.market_spread_home_minus
    raw_edge_toward_dog = max(0.0,  raw_diff)                 # model less pro-home
    raw_edge_toward_fav = max(0.0, -raw_diff)                 # model more pro-home

    # clamp how far we're willing to differ from the book (for blending & sanity)
    raw_diff_clamped = clamp(raw_diff, -SPREAD_NUDGE_CAP, SPREAD_NUDGE_CAP)

    # -------- Blend with market (spread) for scoring/win% --------
    blended_home_spread = si.market_spread_home_minus + (1.0 - WMKT_SPREAD) * raw_diff_clamped

    # -------- Win probability from blended spread --------
    market_prior = logistic(-K_SPREAD * blended_home_spread)
    home_wp = 0.5 + (market_prior - 0.5) * PROB_SQUASH_TEMP
    away_wp = 1 - home_wp

    # -------- Total: tiny model nudge, tightly clamped --------
    total_nudge_raw = clamp(0.03 * abs(power_diff) + 0.05 * (si.home_l5 + si.away_l5),
                            -TOTAL_NUDGE_CAP, TOTAL_NUDGE_CAP)
    blended_total = si.market_total + (1.0 - WMKT_TOTAL) * total_nudge_raw
    blended_total = clamp(blended_total,
                          si.market_total - FINAL_TOTAL_CLAMP,
                          si.market_total + FINAL_TOTAL_CLAMP)

    # -------- Split score from (total, spread) --------
    # home_pts - away_pts = -blended_home_spread
    home_pts = blended_total / 2 - blended_home_spread / 2
    away_pts = blended_total - home_pts

    # Soft caps
    home_pts = clamp(home_pts, 80, 135)
    away_pts = clamp(away_pts, 80, 135)
    tot_after = home_pts + away_pts
    if abs(tot_after - blended_total) > 0.01:
        scale = blended_total / tot_after
        home_pts = clamp(home_pts * scale, 80, 135)
        away_pts = clamp(away_pts * scale, 80, 135)

    # -------- Side recommendation (NFL-style, RAW edge thresholds) --------
    home_is_fav = si.market_spread_home_minus < 0
    fav_team = si.home_team if home_is_fav else si.away_team
    dog_team = si.away_team if home_is_fav else si.home_team
    book_line = si.market_spread_home_minus  # home-spread sign

    side = f"{fav_team} ML"  # default: favorite ML

    # Lay favorite only if RAW edge toward favorite is big enough
    if raw_edge_toward_fav >= EDGE_TO_LAY_FAV:
        side = f"{fav_team} {book_line:.1f}"

    # Else take dog +book if RAW edge toward dog is big enough
    elif raw_edge_toward_dog >= EDGE_TO_TAKE_DOG:
        side = f"{dog_team} +{abs(book_line):.1f}"

    # -------- Totals recommendation --------
    total_edge = blended_total - si.market_total
    if total_edge >= 3.5:
        total_pick = f"Over {si.market_total:.1f}"
    elif total_edge <= -3.5:
        total_pick = f"Under {si.market_total:.1f}"
    else:
        total_pick = "Pass total"

    # -------- Confidence (simple, modest) --------
    base_conf = 52 + 5.0 * np.tanh(abs(blended_home_spread) / 6.0)
    # more confidence when RAW edge vs book is bigger
    raw_edge_points = max(raw_edge_toward_dog, raw_edge_toward_fav)
    base_conf += 4.0 * np.tanh(max(0.0, raw_edge_points - 1.0) / 2.0)
    base_conf = clamp(base_conf, 55, CONF_BASEMAX)

    return {
        "home_win_prob": round(home_wp * 100, 1),
        "away_win_prob": round(away_wp * 100, 1),
        "pred_home": round(home_pts, 1),
        "pred_away": round(away_pts, 1),
        "pred_total": round(blended_total, 1),
        "side_play": side,
        "total_play": total_pick,
        "confidence_pct": round(base_conf, 1),
        "meta": {
            "model_home_spread": round(model_home_spread, 2),
            "book_home_spread": si.market_spread_home_minus,
            "raw_diff_model_minus_book": round(raw_diff, 2),
            "raw_edge_toward_dog": round(raw_edge_toward_dog, 2),
            "raw_edge_toward_fav": round(raw_edge_toward_fav, 2),
            "blended_home_spread": round(blended_home_spread, 2),
            "blended_total": round(blended_total, 2),
            "total_nudge_raw": round(total_nudge_raw, 2),
        }
    }

# =================== UI ===================
if 'slate' not in st.session_state:
    st.session_state.slate: List[Dict] = []

with st.form("game"):
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.text_input("Home Team", "Hawks")
        away_team = st.text_input("Away Team", "Raptors")
        home_power = st.number_input("Home Power", 1200.0, 1900.0, 1600.0, 1.0)
        away_power = st.number_input("Away Power", 1200.0, 1900.0, 1585.0, 1.0)
    with c2:
        market_spread_home_minus = st.number_input("Market Spread (home negative)", -20.0, 20.0, -5.5, 0.5)
        market_total = st.number_input("Market Total", 170.0, 280.0, 230.5, 0.5)
        home_l5 = st.number_input("Home L5 Avg Margin (optional)", -20.0, 20.0, 0.0, 0.1)
        away_l5 = st.number_input("Away L5 Avg Margin (optional)", -20.0, 20.0, 0.0, 0.1)

    submitted = st.form_submit_button("Predict Game")

if submitted:
    si = Inputs(
        home_team=home_team,
        away_team=away_team,
        home_power=home_power,
        away_power=away_power,
        market_spread_home_minus=market_spread_home_minus,
        market_total=market_total,
        home_l5=home_l5,
        away_l5=away_l5
    )
    out = predict(si)

    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{home_team} Win %", f"{out['home_win_prob']}%")
    m2.metric(f"{away_team} Win %", f"{out['away_win_prob']}%")
    m3.metric("Predicted Total", out['pred_total'])
    m4.metric("Confidence", f"{out['confidence_pct']}%")

    st.write(f"**Predicted Score:** {out['pred_away']} @ {out['pred_home']}")
    st.write(f"**Side:** {out['side_play']} | **Total:** {out['total_play']}")

    if SHOW_DEBUG:
        with st.expander("Debug"):
            st.json(out["meta"])

    if st.button("➕ Add to Slate"):
        st.session_state.slate.append({'inputs': asdict(si), 'output': out})
        st.success("Added to slate.")

st.markdown("---")
st.subheader("Tonight's Slate")
if st.session_state.slate:
    df = pd.DataFrame([
        {
            'Matchup': f"{g['inputs']['away_team']} @ {g['inputs']['home_team']}",
            'Pick (Side)': g['output']['side_play'],
            'Total Pick': g['output']['total_play'],
            'Pred Score': f"{g['output']['pred_away']} - {g['output']['pred_home']}",
            'Home Win %': g['output']['home_win_prob'],
            'Conf %': g['output']['confidence_pct'],
        }
        for g in st.session_state.slate
    ])
    st.dataframe(df, use_container_width=True)

    ranked = sorted(
        [g for g in st.session_state.slate if g['output']['confidence_pct'] >= 68],
        key=lambda x: x['output']['confidence_pct'], reverse=True
    )

    st.markdown("### ⭐ Top Plays (Conf ≥ 68%)")
    if ranked:
        for i, g in enumerate(ranked[:3], 1):
            st.write(f"**{i}.** {g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']} ({g['output']['confidence_pct']}% conf)")
    else:
        st.info("No plays ≥ 68% confidence yet.")

    if st.button("🗑️ Clear Slate"):
        st.session_state.slate = []
        st.info("Slate cleared.")
else:
    st.info("Add a game above to build the slate.")
