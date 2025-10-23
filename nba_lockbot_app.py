import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

"""
NBA Prediction — Accuracy-Tuned v2 (Like Our NFL)
Minimal inputs, market-aware logic, disciplined spreads/totals, optional injury tweaks.
"""

# ==== Tunables / Debug (adjust here without hunting through code) ====
SHOW_DEBUG = False            # True to show the debug meta expander
MARKET_WEIGHT = 0.65          # λ: weight on market prior for win% blend
TOTAL_MARKET_WEIGHT = 0.80    # hybrid total = (market*w) + (model*(1-w))
LAY_BUFFER = 2.5              # need to beat book line by this many pts to lay the spread
MIN_CONF_LAY = 72             # base confidence needed to lay the spread
DOG_CONFLICT_BUFFER = 1.5     # extra vs book needed to take dog +spread when we disagree
ROAD_FAV_NOGO = 2.5           # road favorite guardrail threshold (don’t fade lightly)
PACE_MIN, PACE_MAX = 96.0, 102.0
LEAGUE_ORtg = 112.0           # baseline offensive rating per 100

# Streamlit setup
st.set_page_config(page_title="NBA Predictor — Tuned v2", layout="wide")
st.title("🏀 NBA Prediction — Accuracy-Tuned v2 (Like Our NFL)")
st.caption("Minimal inputs. Market-aware edges, disciplined spread usage, safer totals, optional advanced tweaks.")

# ---------------------------
# Helpers
# ---------------------------
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
    market_spread_home_minus: float  # home favored = negative (e.g., -4.5); positive => away favored
    market_total: float
    # Optional advanced adjustments (per 100)
    home_off_missing: float = 0.0
    home_def_missing: float = 0.0
    away_off_missing: float = 0.0
    away_def_missing: float = 0.0

def predict(si: SimpleInputs) -> Dict:
    # Injury-adjusted nets (offense down, defense worse -> subtract)
    adj_home_net = (si.home_net - si.home_off_missing) - si.home_def_missing
    adj_away_net = (si.away_net - si.away_off_missing) - si.away_def_missing

    elo_diff = si.home_elo - si.away_elo
    net_diff = adj_home_net - adj_away_net
    form_diff = si.home_l5 - si.away_l5

    # Compact model → home win prob
    w_hca, w_elo, w_net, w_form = 0.5, 0.008, 0.080, 0.055
    linear = (w_hca + w_elo*elo_diff + w_net*net_diff + w_form*form_diff)
    home_wp_model = logistic(linear)

    # Market prior from spread (home negative => favored)
    k = 0.24
    market_prior = logistic(-k * si.market_spread_home_minus)

    # Blend model with market prior
    lam = MARKET_WEIGHT
    home_wp = (1 - lam) * home_wp_model + lam * market_prior
    away_wp = 1 - home_wp

    # Auto pace (no input), clamped
    pace_est = 99.5 + 0.18 * (adj_home_net + adj_away_net) + 0.08 * (si.home_l5 + si.away_l5)
    pace_est = clamp(pace_est, PACE_MIN, PACE_MAX)

    # Model total, then hybrid with market
    net_nudge = 0.35 * (adj_home_net + adj_away_net)
    form_nudge = 0.25 * (si.home_l5 + si.away_l5)
    model_total_per100 = (LEAGUE_ORtg * 2) + net_nudge + form_nudge
    model_total = clamp(pace_est * model_total_per100 / 100.0, 180, 270)
    pred_total = TOTAL_MARKET_WEIGHT * si.market_total + (1 - TOTAL_MARKET_WEIGHT) * model_total

    # Expected margin (home + = home by X)
    exp_margin = (home_wp - 0.5) * 20 + 0.42 * net_diff + 0.25 * form_diff

    # Split points
    share = clamp(0.5 + 0.25 * (net_diff / 20.0), 0.35, 0.65)
    home_pts = clamp(pred_total * share + 0.4 * exp_margin, 70, 160)
    away_pts = clamp(pred_total - home_pts, 70, 160)

    # ---------- Confidence (base) ----------
    total_edge = pred_total - si.market_total
    base_components = [
        min(abs(elo_diff)/50.0, 1.2),
        min(abs(net_diff)/6.0, 1.2),
        min(abs(form_diff)/6.0, 1.0),
        min(abs(total_edge)/8.0, 1.0) * 0.4,
    ]
    base_conf = clamp(48 + 34 * np.tanh(sum(base_components)), 50, 90)

    # ---------- Side pick (market-aware) ----------
    home_is_market_fav = si.market_spread_home_minus < 0
    market_home_fav_by = -si.market_spread_home_minus   # +X => home -X,  -Y => away -Y
    home_fav_by = max(0.0, market_home_fav_by)
    away_fav_by = max(0.0, -market_home_fav_by)
    model_home_spread = -exp_margin  # negative => our model favors home

    # Start on market favorite ML
    side = f"{si.home_team} ML" if home_is_market_fav else f"{si.away_team} ML"

    # If we align with market favorite and beat the line by LAY_BUFFER with enough confidence → lay book line
    if home_is_market_fav and (exp_margin >= home_fav_by + LAY_BUFFER) and base_conf >= MIN_CONF_LAY:
        side = f"{si.home_team} -{home_fav_by:.1f}"
    elif (not home_is_market_fav) and ((-exp_margin) >= away_fav_by + LAY_BUFFER) and base_conf >= MIN_CONF_LAY:
        side = f"{si.away_team} -{away_fav_by:.1f}"

    # Strong disagreement → prefer dog +spread (safer) if stronger than book by DOG_CONFLICT_BUFFER
    disagree = (home_is_market_fav and model_home_spread > 0) or ((not home_is_market_fav) and model_home_spread < 0)
    if disagree and base_conf >= max(60, MIN_CONF_LAY - 2):
        book_line = home_fav_by if home_is_market_fav else away_fav_by
        if abs(model_home_spread) >= (book_line + DOG_CONFLICT_BUFFER):
            side = (f"{si.away_team} +{home_fav_by:.1f}" if home_is_market_fav
                    else f"{si.home_team} +{away_fav_by:.1f}")

    # Road favorite guardrail: if away is -ROAD_FAV_NOGO+ and our home win% < 58%, avoid fading
    if away_fav_by >= ROAD_FAV_NOGO and home_wp < 0.58:
        if ((-exp_margin) >= away_fav_by + LAY_BUFFER) and base_conf >= MIN_CONF_LAY:
            side = f"{si.away_team} -{away_fav_by:.1f}"
        else:
            side = f"{si.away_team} ML"

    # Tight 47–53 → give market dog +1.5 if we’re still on ML
    if 0.47 <= home_wp <= 0.53 and ('+' not in side and '-' not in side):
        side = f"{si.home_team} +1.5" if not home_is_market_fav else f"{si.away_team} +1.5"

    # Confidence shaping
    pick_is_home = side.startswith(si.home_team)
    fading_market_fav = (home_is_market_fav and not pick_is_home) or ((not home_is_market_fav) and pick_is_home)
    conf = base_conf - (4.0 if fading_market_fav else 0.0) + (2.0 if ('-' in side) else 0.0)
    final_conf = clamp(conf, 50, 90)

    # ---------- Totals (hybrid) ----------
    if (pred_total - si.market_total) >= 5.0:
        total_pick = f"Over {si.market_total:.1f}"
    elif (pred_total - si.market_total) <= -5.0:
        total_pick = f"Under {si.market_total:.1f}"
    else:
        total_pick = "Pass total"

    return {
        "home_win_prob": round(100 * home_wp, 1),
        "away_win_prob": round(100 * away_wp, 1),
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
            "model_home_spread": round(model_home_spread, 2),
            "conf_base": round(base_conf, 1),
            "pred_total_model": round(model_total, 1),
            "pred_total_market": round(si.market_total, 1),
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

    with st.expander("Advanced (optional)"):
        c3, c4 = st.columns(2)
        with c3:
            home_off_missing = st.number_input("Home Off Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            home_def_missing = st.number_input("Home Def Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
        with c4:
            away_off_missing = st.number_input("Away Off Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            away_def_missing = st.number_input("Away Def Missing (per 100)", 0.0, 20.0, 0.0, 0.1)

    submitted = st.form_submit_button("Predict Game")

if submitted:
    si = SimpleInputs(
        home_team, away_team, home_net, away_net, home_l5, away_l5,
        home_elo, away_elo, market_spread_home_minus, market_total,
        home_off_missing, home_def_missing, away_off_missing, away_def_missing
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

    if SHOW_DEBUG:
        with st.expander("Debug meta (for tuning)"):
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

    # High-confidence plays only
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

    # 🔒 Lock: ≥72% and cannot fade a road favorite of ROAD_FAV_NOGO+
    lock_candidates = []
    for g in ranked:
        meta = g['output'].get('meta', {})
        away_fav_by = meta.get('away_fav_by', 0.0)
        pick = g['output']['side_play']
        pick_is_home = pick.startswith(g['inputs']['home_team'])
        fades_road_fav = (away_fav_by >= ROAD_FAV_NOGO) and pick_is_home
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
