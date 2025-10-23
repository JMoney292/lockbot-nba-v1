import math
from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

"""
NBA Predictor — NFL-Style Minimal (v1.1)
- Market-led: spread & total are the truth; we add tiny power/form nudges.
- Convert blended spread -> win% with calibrated curve (NBA: -5.5 ≈ ~70% ML).
- Split team scores directly from (total, spread). No pace, no extras.
- FIXED: underdog +spread logic (will now give the dog +book when our number
  is ≥ EDGE_TO_LAY points toward the dog).
"""

# =================== Tunables (keep tiny) ===================
SHOW_DEBUG = False

# Calibrated spread -> win% (NBA)
K_SPREAD = 0.15             # => -5.5 ~70% ML
PROB_SQUASH_TEMP = 0.90     # compress probabilities toward 0.5

# Market anchoring
WMKT_SPREAD = 0.85          # 85% market, 15% model for spread
WMKT_TOTAL  = 0.90          # 90% market, 10% model for total

# Tiny model components (small, stable)
HFA_PTS = 1.8               # baseline home-court (pts)
BETA_POWER = 0.06           # pts per (home_power - away_power)
BETA_FORM  = 0.04           # pts per (homeL5 - awayL5)

# Caps (keep model in its lane)
SPREAD_NUDGE_CAP = 1.5      # our model can move spread by at most ±1.5 vs book
TOTAL_NUDGE_CAP  = 2.0      # our model can move total by at most ±2.0 vs book
FINAL_TOTAL_CLAMP = 4.0     # final total stays within ±4 of market

# Recommendation thresholds
EDGE_TO_LAY  = 2.5          # need to beat book spread by ≥2.5 to lay the number
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
    home_power: float    # ELO/power, any scale (relative diffs matter)
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
    model_margin = clamp(model_margin, -8, 8)  # safety cap

    # Convert to home-spread sign convention (negative = home favored)
    model_home_spread = -model_margin

    # -------- Blend with market (spread) --------
    # Amount we would move the book line (clamped small)
    raw_spread_nudge = model_home_spread - si.market_spread_home_minus
    spread_nudge = clamp(raw_spread_nudge, -SPREAD_NUDGE_CAP, SPREAD_NUDGE_CAP)
    blended_home_spread = si.market_spread_home_minus + (1.0 - WMKT_SPREAD) * spread_nudge

    # -------- Win probability from blended spread --------
    market_prior = logistic(-K_SPREAD * blended_home_spread)
    home_wp = 0.5 + (market_prior - 0.5) * PROB_SQUASH_TEMP
    away_wp = 1 - home_wp

    # -------- Total: tiny model nudge, tightly clamped --------
    total_nudge = clamp(0.03 * abs(power_diff) + 0.05 * (si.home_l5 + si.away_l5),
                        -TOTAL_NUDGE_CAP, TOTAL_NUDGE_CAP)
    blended_total = si.market_total + (1.0 - WMKT_TOTAL) * total_nudge
    blended_total = clamp(blended_total,
                          si.market_total - FINAL_TOTAL_CLAMP,
                          si.market_total + FINAL_TOTAL_CLAMP)

    # -------- Split score from (total, spread) --------
    # home_pts - away_pts = -blended_home_spread
    home_pts = blended_total / 2 - blended_home_spread / 2
    away_pts = blended_total - home_pts

    # Soft caps for realism
    home_pts = clamp(home_pts, 80, 135)
    away_pts = clamp(away_pts, 80, 135)
    tot_after = home_pts + away_pts
    if abs(tot_after - blended_total) > 0.01:
        scale = blended_total / tot_after
        home_pts = clamp(home_pts * scale, 80, 135)
        away_pts = clamp(away_pts * scale, 80, 135)

    # -------- Side recommendation (NFL-style, FIXED dog logic) --------
    home_is_fav = si.market_spread_home_minus < 0
    fav_team = si.home_team if home_is_fav else si.away_team
    dog_team = si.away_team if home_is_fav else si.home_team
    book_line = si.market_spread_home_minus  # home-spread sign

    # Edges relative to book, measured in points toward fav or dog
    # If blended number is less pro-home than book, that's "toward dog".
    edge_toward_dog = blended_home_spread - book_line
    edge_toward_fav = -edge_toward_dog

    # Default: ML on favorite
    side = f"{fav_team} ML"

    # Lay the book line only if we beat it toward the favorite by EDGE_TO_LAY+
    if edge_toward_fav >= EDGE_TO_LAY:
        side = f"{fav_team} {book_line:.1f}"

    # Otherwise, if we beat it toward the dog by EDGE_TO_LAY+, take dog +spread
    elif edge_toward_dog >= EDGE_TO_LAY:
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
    base_conf = 52 + 5.0 * np.tanh(abs(blended_home_spread) / 6.0) * 100/100
    # extra confidence if we pushed hard against the book (beyond 0.5 of our cap)
    beat_book = max(0.0, abs(spread_nudge) - 0.5) / (SPREAD_NUDGE_CAP)
    base_conf += 8.0 * beat_book
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
            "blended_home_spread": round(blended_home_spread, 2),
            "spread_nudge": round(spread_nudge, 2),
            "blended_total": round(blended_total, 2),
            "total_nudge": round(total_nudge, 2),
            "market_spread": si.market_spread_home_minus,
            "market_total": si.market_total,
            "edge_toward_dog": round(edge_toward_dog, 2),
            "edge_toward_fav": round(edge_toward_fav, 2),
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
