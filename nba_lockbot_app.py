import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

"""
NBA Prediction — **Accuracy-Tuned Simple Model (Like Our NFL)**

Upgrades applied:
- Market prior blend 45% (respects sharp spreads)
- Lower home-court weight (road favorites won’t get drowned)
- Road-favorite guardrail (don’t fade away -2.5+ lightly)
- Only lay full market line when our edge clearly beats the book (+2 pts) AND confidence ≥ 68%
- Totals recommended only with ≥ 5.0-pt edge
- Auto pace (no input), clamped tighter (96–102)
- Confidence penalty for fading market favorite
- Top 3 require ≥ 68% confidence; 🔒 Lock requires ≥ 72% and cannot fade a road favorite
"""

st.set_page_config(page_title="NBA Predictor — Tuned", layout="wide")
st.title("🏀 NBA Prediction — Accuracy-Tuned (Like Our NFL)")
st.caption("Minimal inputs. Market-aware edges, safer spread usage, better totals filter, and stricter Lock rules.")

# ---------------------------
# Helpers
# ---------------------------
LEAGUE_ORtg = 112.0  # baseline per 100


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class SimpleInputs:
    home_team: str
    away_team: str
    home_net: float      # Net Rating (per 100)
    away_net: float
    home_l5: float       # last-5 avg margin
    away_l5: float
    home_elo: float
    away_elo: float
    market_spread_home_minus: float  # home favored = negative
    market_total: float


def predict(si: SimpleInputs) -> Dict:
    elo_diff = si.home_elo - si.away_elo
    net_diff = si.home_net - si.away_net
    form_diff = si.home_l5 - si.away_l5

    # Linear score → win prob (compact model)
    w_hca = 0.5  # further reduced home-court          # reduced home-court
    w_elo = 0.008        # per elo point
    w_net = 0.075        # per net rating point
    w_form = 0.06        # per margin point

    linear = (w_hca + w_elo*elo_diff + w_net*net_diff + w_form*form_diff)
    home_wp_model = logistic(linear)

    # --- Market-aware prior from spread (home negative => favored)
    k = 0.23  # mapping spread→prob slope
    market_prior = logistic(-k * si.market_spread_home_minus)

    # Blend model with market prior (lam = weight on market)
    lam = 0.60
    home_wp = (1 - lam) * home_wp_model + lam * market_prior
    away_wp = 1 - home_wp

    # --- Auto-estimated pace (no input) ---
    pace_est = 99.5 + 0.20 * (si.home_net + si.away_net) + 0.10 * (si.home_l5 + si.away_l5)
    pace_est = clamp(pace_est, 96.0, 102.0)

    # Predicted total using pace_est and net/form
    net_nudge = 0.35 * (si.home_net + si.away_net)
    form_nudge = 0.25 * (si.home_l5 + si.away_l5)
    model_total_per100 = (LEAGUE_ORtg*2) + net_nudge + form_nudge  # per 100 poss
    model_total = clamp(pace_est * model_total_per100 / 100.0, 180, 270)

    # Hybrid to respect the book: 70% market, 30% model
    pred_total = 0.70 * si.market_total + 0.30 * model_total

    # Expected margin
    exp_margin = (home_wp - 0.5) * 20 + 0.4 * net_diff + 0.25 * form_diff

    # Split total into team points by relative net strength
    share = clamp(0.5 + 0.25 * (net_diff / 20.0), 0.35, 0.65)
    home_pts = clamp(pred_total * share + 0.4*exp_margin, 70, 160)
    away_pts = clamp(pred_total - home_pts, 70, 160)

    # ---------------------------
    # Confidence (base)
    # ---------------------------
    total_edge = pred_total - si.market_total
    base_components = [
        min(abs(elo_diff)/50.0, 1.2),
        min(abs(net_diff)/6.0, 1.2),
        min(abs(form_diff)/6.0, 1.0),
        min(abs(total_edge)/8.0, 1.0) * 0.5,
    ]
    base_conf = clamp(48 + 34*np.tanh(sum(base_components)), 50, 90)

    # ---------------------------
    # Side pick — stronger market anchoring
    # ---------------------------
    favor_home = home_wp >= 0.5
    home_is_market_fav = si.market_spread_home_minus < 0
    market_home_fav_by = -si.market_spread_home_minus      # +X => home -X,  -Y => away -Y
    home_fav_by = max(0.0, market_home_fav_by)
    away_fav_by = max(0.0, -market_home_fav_by)

    # Default side follows the market favorite unless our model shows a clear >2.0 pt edge the other way
    market_fav_team = 'home' if home_is_market_fav else 'away'

    # Model-implied spread for home (positive = we think home should be favored)
    model_spread_home = -exp_margin  # if home expected_margin>0, home should win; convert to away-home sign

    # If model conflicts with market but by less than (market line + 1), stick with market
    conflict_with_market = (home_is_market_fav and model_spread_home > 0) or ((not home_is_market_fav) and model_spread_home < 0)
    strong_conflict = False
    if conflict_with_market:
        needed = (home_fav_by if home_is_market_fav else away_fav_by) + 1.0
        strong_conflict = abs(model_spread_home) >= needed

    # Start with market favorite ML
    if market_fav_team == 'home':
        side = f"{si.home_team} ML"
    else:
        side = f"{si.away_team} ML"

    # If we align with the market favorite and beat the number by +2 and conf ≥ 70, lay the market line
    if home_is_market_fav and (exp_margin >= home_fav_by + 2) and base_conf >= 70:
        side = f"{si.home_team} -{home_fav_by:.1f}"
    elif (not home_is_market_fav) and ((-exp_margin) >= away_fav_by + 2) and base_conf >= 70:
        side = f"{si.away_team} -{away_fav_by:.1f}"

    # If we strongly disagree with market, choose the dog +spread instead of ML (safer), require strong_conflict
    if strong_conflict and base_conf >= 68:
        if home_is_market_fav:
            # Market says home -, we like away -> take Away +line
            side = f"{si.away_team} +{home_fav_by:.1f}"
        else:
            side = f"{si.home_team} +{away_fav_by:.1f}"

    # Road favorite guardrail: if away is -2.5+ and our home win% < 58%, avoid fading
    if away_fav_by >= 2.5 and home_wp < 0.58:
        if ((-exp_margin) >= away_fav_by + 2) and base_conf >= 70:
            side = f"{si.away_team} -{away_fav_by:.1f}"
        else:
            side = f"{si.away_team} ML"

    # Tight-game safety 47–53: give market underdog +1.5 (only if not already using market +spread)
    if 0.47 <= home_wp <= 0.53 and ('+' not in side and '-' not in side):
        side = f"{si.home_team} +1.5" if not home_is_market_fav else f"{si.away_team} +1.5"

    # Confidence penalty if fading market favorite
    pick_is_home = side.startswith(si.home_team)
    we_fade_market_fav = (home_is_market_fav and not pick_is_home) or ((not home_is_market_fav) and pick_is_home)
    conf_penalty = 5.0 if we_fade_market_fav else 0.0

    # ---------------------------
    # Total pick — stricter (using hybrid total)
    # ---------------------------
    # ---------------------------
    if (pred_total - si.market_total) >= 5.0:
        total_pick = f"Over {si.market_total:.1f}"
    elif (pred_total - si.market_total) <= -5.0:
        total_pick = f"Under {si.market_total:.1f}"
    else:
        total_pick = "Pass total"

    final_conf = clamp(base_conf - conf_penalty, 50, 90)

    return {
        "home_win_prob": round(100*home_wp, 1),
        "away_win_prob": round(100*away_wp, 1),
        "pred_home": round(home_pts, 1),
        "pred_away": round(away_pts, 1),
        "pred_total": round(pred_total, 1),
        "expected_margin": round(exp_margin, 2),
        "side_play": side,
        "total_play": total_pick,
        "confidence_pct": round(final_conf, 1),
        "meta": {
            "home_is_market_fav": home_is_market_fav,
            "home_fav_by": home_fav_by,
            "away_fav_by": away_fav_by,
            "conf_base": round(base_conf, 1),
            "conf_penalty": conf_penalty,
            "total_edge": round(pred_total - si.market_total, 1),
        }
    }

# ---------------------------
# UI
# ---------------------------
if 'slate' not in st.session_state:
    st.session_state.slate: List[Dict] = []

with st.form("simple_game"):
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.text_input("Home Team", "Lakers")
        away_team = st.text_input("Away Team", "Warriors")
        home_net = st.number_input("Home Net Rating", -20.0, 20.0, 3.0, 0.1)
        away_net = st.number_input("Away Net Rating", -20.0, 20.0, 1.0, 0.1)
        home_l5 = st.number_input("Home L5 Avg Margin", -20.0, 20.0, 2.0, 0.1)
        away_l5 = st.number_input("Away L5 Avg Margin", -20.0, 20.0, 0.0, 0.1)
    with c2:
        home_elo = st.number_input("Home Power/ELO", 1300.0, 1900.0, 1650.0, 1.0)
        away_elo = st.number_input("Away Power/ELO", 1300.0, 1900.0, 1635.0, 1.0)
        market_spread_home_minus = st.number_input("Market Spread (home negative)", -20.0, 20.0, -3.5, 0.5)
        market_total = st.number_input("Market Total", 170.0, 280.0, 228.5, 0.5)

    submitted = st.form_submit_button("Predict Game")

if submitted:
    si = SimpleInputs(
        home_team, away_team, home_net, away_net, home_l5, away_l5,
        home_elo, away_elo, market_spread_home_minus, market_total
    )
    out = predict(si)

    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{home_team} Win %", f"{out['home_win_prob']}%")
    m2.metric(f"{away_team} Win %", f"{out['away_win_prob']}%")
    m3.metric("Predicted Total", out['pred_total'])
    m4.metric("Confidence", f"{out['confidence_pct']}%")

    st.write(f"**Predicted Score:** {away_team} {out['pred_away']} @ {home_team} {out['pred_home']}")
    st.write(f"**Side:** {out['side_play']}  |  **Total:** {out['total_play']}  |  **Edge (spread est):** {out['expected_margin']} pts")
    with st.expander("Debug meta (for tuning)"):
        st.json(out["meta"])

    if st.button("➕ Add to Slate"):
        st.session_state.slate.append({
            'inputs': asdict(si),
            'output': out
        })
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

    # Filter for Top 3: only high-confidence picks
    ranked = sorted(
        [g for g in st.session_state.slate if g['output']['confidence_pct'] >= 68],
        key=lambda x: x['output']['confidence_pct'], reverse=True
    )

    st.markdown("### ⭐ Top Plays (Conf ≥ 68%)")
    if ranked:
        for i, g in enumerate(ranked[:3], 1):
            st.write(f"**{i}.** {g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']} ({g['output']['confidence_pct']}% conf)")
    else:
        st.info("No plays ≥ 68% confidence.")

    # 🔒 Lock: must be ≥72% and cannot fade a road favorite
    lock_candidates = []
    for g in ranked:
        meta = g['output'].get('meta', {})
        home_is_fav = meta.get('home_is_market_fav', False)
        away_fav_by = meta.get('away_fav_by', 0.0)
        pick = g['output']['side_play']
        pick_is_home = pick.startswith(g['inputs']['home_team'])
        fades_road_fav = (away_fav_by >= 2.5) and pick_is_home
        if (g['output']['confidence_pct'] >= 72) and (not fades_road_fav):
            lock_candidates.append(g)
    if lock_candidates:
        g = lock_candidates[0]
        st.markdown("### 🔒 Lock of the Day")
        st.success(f"{g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']}  ({g['output']['confidence_pct']}% confidence)")
    else:
        st.info("No safe Lock meeting the ≥72% & no-road-fade rules.")

    if st.button("🗑️ Clear Slate"):
        st.session_state.slate = []
        st.info("Slate cleared.")
else:
    st.info("Add a game result above to build the slate and get Top 3 + 🔒 Lock.")

st.markdown("""
**Inputs**
- **Net Rating:** ORtg − DRtg (per 100). Overall net works if you don’t have home/away splits.
- **L5 Avg Margin:** Avg point diff over last 5.
- **Power/ELO:** Any 1500–1800-style number; higher = better.
- **Market Spread:** Negative if home is favored (e.g., -4.5). Positive if home is a dog (e.g., +2.5 means away favored by 2.5).
- **Market Total:** Book total.

**How picks are chosen**
- Moneyline by default on the side with higher win %.
- Lay the **market line** only if we beat it by **≥ 2 pts** and **confidence ≥ 68%**.
- Don’t fade road favorites of **≥ 2.5** unless our edge is big; otherwise lean Away ML.
- Totals only when **edge ≥ 5.0**.
- Confidence is trimmed when fading the market favorite.
""")
