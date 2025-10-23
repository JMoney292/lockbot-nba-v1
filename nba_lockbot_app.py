import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

"""
NBA Prediction — Accuracy-Tuned v4.1 (Conservative + Calibrated)
- Market-anchored totals with hard ±5 clamp
- Narrow pace band (95.5–100.5)
- Smaller net/form nudges into totals
- Blowout dampener on totals for big lines
- Gentler score split and tighter team point caps
- Disciplined, market-aware side logic
- ***New calibration***: spread→win% fixed (e.g., ~70% at -5.5, not ~80%),
  softer market weight, probability squashing, and confidence cap on mid lines
"""

# ==== Tunables / Debug ====
SHOW_DEBUG = False

# Win% blend (dynamic by line; floors/ceils applied inside)
MARKET_WEIGHT_MIN = 0.58
MARKET_WEIGHT_MAX = 0.75

# Spread→prob slope (calibrated so -5.5 ≈ ~70% ML)
K_SPREAD = 0.15         # was ~0.21–0.26; lower is safer/realistic
PROB_SQUASH_TEMP = 0.88 # compresses probabilities toward 0.5 to avoid extremes

# Totals
TOTAL_CLAMP_ABS = 5.0         # final total cannot deviate from book by more than ±5
TOTAL_ALPHA_MIN = 0.06        # shrinkage from model toward market on big lines
TOTAL_ALPHA_MAX = 0.12        # shrinkage on small lines
TOTAL_EDGE_FIRE = 6.0         # only fire O/U if |edge| >= this

# Last-5 team totals nudge (tiny; set TT_BETA=0 to disable)
TT_BETA = 0.10
TT_CLAMP = 4.0

# Pace
PACE_MIN, PACE_MAX = 95.5, 100.5
LEAGUE_ORtg = 112.0

# Spread discipline (sides)
LAY_BUFFER = 3.0
MIN_CONF_LAY = 74
DOG_CONFLICT_BUFFER = 2.5
ROAD_FAV_NOGO1 = 2.0
ROAD_FAV_NOGO2 = 3.5
ROAD_FAV_NOGO3 = 6.0

st.set_page_config(page_title="NBA Predictor — Tuned v4.1", layout="wide")
st.title("🏀 NBA Prediction — Accuracy-Tuned v4.1")
st.caption("Conservative totals + calibrated win%: safer scores, fewer runaway favorites, and disciplined spreads/totals.")

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
    home_net: float
    away_net: float
    home_l5: float
    away_l5: float
    home_elo: float
    away_elo: float
    market_spread_home_minus: float  # home favored = negative
    market_total: float
    # Optional adjustments (per 100)
    home_off_missing: float = 0.0
    home_def_missing: float = 0.0
    away_off_missing: float = 0.0
    away_def_missing: float = 0.0
    # Optional last-5 team totals (raw points)
    home_l5_team_total: Optional[float] = None
    away_l5_team_total: Optional[float] = None

def predict(si: SimpleInputs) -> Dict:
    # Injury-adjusted nets
    adj_home_net = (si.home_net - si.home_off_missing) - si.home_def_missing
    adj_away_net = (si.away_net - si.away_off_missing) - si.away_def_missing

    elo_diff  = si.home_elo - si.away_elo
    net_diff  = adj_home_net - adj_away_net
    form_diff = si.home_l5 - si.away_l5

    # --- Home win prob (calibrated + squashed) ---
    w_hca, w_elo, w_net, w_form = 0.5, 0.008, 0.080, 0.055
    linear = (w_hca + w_elo*elo_diff + w_net*net_diff + w_form*form_diff)
    home_wp_model = logistic(linear)

    abs_line = abs(si.market_spread_home_minus)

    # Market prior from spread with calibrated slope (e.g., -5.5 ≈ ~70% ML)
    market_prior = logistic(-K_SPREAD * si.market_spread_home_minus)

    # Softer market weight scales with line size (max ~0.75)
    lam = clamp(0.58 + 0.02 * min(abs_line, 10), MARKET_WEIGHT_MIN, MARKET_WEIGHT_MAX)

    # Blend and then compress probabilities toward 0.5 to avoid extremes
    home_wp = (1 - lam) * home_wp_model + lam * market_prior
    home_wp = 0.5 + (home_wp - 0.5) * PROB_SQUASH_TEMP
    away_wp = 1 - home_wp

    # --- Pace (narrow and conservative) ---
    pace_est = 99.2 + 0.12 * (adj_home_net + adj_away_net) + 0.05 * (si.home_l5 + si.away_l5)
    if abs(adj_home_net - adj_away_net) > 6:
        pace_est -= 0.4  # mismatches drift slower
    pace_est = clamp(pace_est, PACE_MIN, PACE_MAX)

    # --- Model total with smaller nudges ---
    net_nudge  = 0.20 * (adj_home_net + adj_away_net)   # reduced
    form_nudge = 0.15 * (si.home_l5 + si.away_l5)       # reduced
    model_total_per100 = (LEAGUE_ORtg * 2) + net_nudge + form_nudge
    model_total = clamp(pace_est * model_total_per100 / 100.0, 180, 270)

    # --- Strong shrink toward the market total + hard clamp ---
    # alpha shrinks as line grows (more trust in book on big lines)
    alpha = TOTAL_ALPHA_MAX - (TOTAL_ALPHA_MAX - TOTAL_ALPHA_MIN) * min(abs_line, 10) / 10.0
    pred_total = si.market_total + alpha * (model_total - si.market_total)

    # blowout dampener (large favorites can depress totals late)
    pred_total -= 0.20 * max(0.0, abs_line - 7.0)

    # final clamp to ±TOTAL_CLAMP_ABS from market
    pred_total = clamp(pred_total, si.market_total - TOTAL_CLAMP_ABS, si.market_total + TOTAL_CLAMP_ABS)

    # --- Optional last-5 team totals nudge (very small) ---
    bias_home = bias_away = 0.0
    if TT_BETA > 0:
        market_home_tt = si.market_total/2 - (si.market_spread_home_minus)/2
        market_away_tt = si.market_total - market_home_tt
        if si.home_l5_team_total and si.home_l5_team_total > 0:
            bias_home = clamp(si.home_l5_team_total - market_home_tt, -TT_CLAMP, TT_CLAMP)
        if si.away_l5_team_total and si.away_l5_team_total > 0:
            bias_away = clamp(si.away_l5_team_total - market_away_tt, -TT_CLAMP, TT_CLAMP)
        pred_total = clamp(pred_total + TT_BETA * (bias_home + bias_away),
                           si.market_total - TOTAL_CLAMP_ABS,
                           si.market_total + TOTAL_CLAMP_ABS)

    # --- Expected margin (home + = home by X) ---
    exp_margin = (home_wp - 0.5) * 20 + 0.42 * net_diff + 0.25 * form_diff

    # --- Split total into team scores (gentle, conservative) ---
    share = logistic(exp_margin / 10.0)     # margin→prob curve
    share = clamp(share, 0.45, 0.55)        # keep splits modest
    share += 0.0015 * (bias_home - bias_away)
    share = clamp(share, 0.44, 0.56)

    home_pts = pred_total * share
    away_pts = pred_total - home_pts

    # Safer caps on team points (prevents weird 150+ outputs)
    home_pts = clamp(home_pts, 80.0, 135.0)
    away_pts = clamp(away_pts, 80.0, 135.0)
    # Re-balance to preserve total if clamped
    total_after_clamp = home_pts + away_pts
    if total_after_clamp != pred_total:
        scale = pred_total / total_after_clamp
        home_pts = clamp(home_pts * scale, 80.0, 135.0)
        away_pts = clamp(away_pts * scale, 80.0, 135.0)

    # ---------- Confidence (base) ----------
    total_edge = pred_total - si.market_total
    base_components = [
        min(abs(elo_diff)/50.0, 1.2),
        min(abs(net_diff)/6.0, 1.2),
        min(abs(form_diff)/6.0, 1.0),
        min(abs(total_edge)/8.0, 1.0) * 0.35,  # less emphasis on totals for confidence
    ]
    base_conf = clamp(48 + 34 * np.tanh(sum(base_components)), 50, 90)

    # Cap confidence a bit on mid-range lines (3–7) to avoid 80% “locks” at -5.5 type games
    if 3.0 <= abs_line <= 7.0:
        base_conf = min(base_conf, 76.0)

    # ---------- Side pick (disciplined, market-aware) ----------
    home_is_market_fav = si.market_spread_home_minus < 0
    market_home_fav_by = -si.market_spread_home_minus
    home_fav_by = max(0.0, market_home_fav_by)
    away_fav_by = max(0.0, -market_home_fav_by)
    model_home_spread = -exp_margin  # negative => model favors home

    # Start on market favorite ML
    side = f"{si.home_team} ML" if home_is_market_fav else f"{si.away_team} ML"

    # Road favorite "don't fade" tiers
    if away_fav_by >= ROAD_FAV_NOGO3 and home_wp < 0.62:
        side = f"{si.away_team} ML"
    elif away_fav_by >= ROAD_FAV_NOGO2 and home_wp < 0.60:
        side = f"{si.away_team} ML"
    elif away_fav_by >= ROAD_FAV_NOGO1 and home_wp < 0.58:
        side = f"{si.away_team} ML"

    # Lay book line only with real edge + confidence
    if home_is_market_fav and (exp_margin >= home_fav_by + LAY_BUFFER) and base_conf >= MIN_CONF_LAY:
        side = f"{si.home_team} -{home_fav_by:.1f}"
    elif (not home_is_market_fav) and ((-exp_margin) >= away_fav_by + LAY_BUFFER) and base_conf >= MIN_CONF_LAY:
        side = f"{si.away_team} -{away_fav_by:.1f}"

    # Strong disagreement → prefer dog +spread
    disagree = (home_is_market_fav and model_home_spread > 0) or ((not home_is_market_fav) and model_home_spread < 0)
    if disagree and base_conf >= (MIN_CONF_LAY - 2):
        book_line = home_fav_by if home_is_market_fav else away_fav_by
        if abs(model_home_spread) >= (book_line + DOG_CONFLICT_BUFFER):
            side = (f"{si.away_team} +{home_fav_by:.1f}" if home_is_market_fav
                    else f"{si.home_team} +{away_fav_by:.1f}")

    # Tight 47–53 → market dog +1.5 if still ML
    if 0.47 <= home_wp <= 0.53 and ('+' not in side and '-' not in side):
        side = f"{si.home_team} +1.5" if not home_is_market_fav else f"{si.away_team} +1.5"

    # Confidence shaping
    pick_is_home = side.startswith(si.home_team)
    fading_market_fav = (home_is_market_fav and not pick_is_home) or ((not home_is_market_fav) and pick_is_home)
    conf = base_conf - (6.0 if fading_market_fav else 0.0) + (2.0 if ('-' in side) else 0.0)
    final_conf = clamp(conf, 50, 90)

    # ---------- Totals pick (strict) ----------
    if (pred_total - si.market_total) >= TOTAL_EDGE_FIRE:
        total_pick = f"Over {si.market_total:.1f}"
    elif (pred_total - si.market_total) <= -TOTAL_EDGE_FIRE:
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
            "lam": round(lam, 2),
            "k_spread": round(K_SPREAD, 3),
            "prob_squash_temp": PROB_SQUASH_TEMP,
            "home_is_market_fav": home_is_market_fav,
            "home_fav_by": home_fav_by,
            "away_fav_by": away_fav_by,
            "model_home_spread": round(model_home_spread, 2),
            "conf_base": round(base_conf, 1),
            "alpha_total": round(TOTAL_ALPHA_MAX - (TOTAL_ALPHA_MAX - TOTAL_ALPHA_MIN) * min(abs_line, 10) / 10.0, 3),
            "total_edge": round(pred_total - si.market_total, 1),
            "bias_home_L5TT": round(bias_home, 1),
            "bias_away_L5TT": round(bias_away, 1),
            "pace_est": round(pace_est, 1),
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
        home_team = st.text_input("Home Team", "Hawks")
        away_team = st.text_input("Away Team", "Raptors")
        home_net = st.number_input("Home Net Rating", -20.0, 20.0, 3.0, 0.1)
        away_net = st.number_input("Away Net Rating", -20.0, 20.0, 1.0, 0.1)
        home_l5 = st.number_input("Home L5 Avg Margin", -20.0, 20.0, 2.0, 0.1)
        away_l5 = st.number_input("Away L5 Avg Margin", -20.0, 20.0, 0.0, 0.1)
    with c2:
        home_elo = st.number_input("Home Power/ELO", 1300.0, 1900.0, 1650.0, 1.0)
        away_elo = st.number_input("Away Power/ELO", 1300.0, 1900.0, 1635.0, 1.0)
        market_spread_home_minus = st.number_input("Market Spread (home negative)", -20.0, 20.0, -5.5, 0.5)
        market_total = st.number_input("Market Total", 170.0, 280.0, 231.5, 0.5)

    with st.expander("Advanced (optional)"):
        c3, c4 = st.columns(2)
        with c3:
            home_off_missing = st.number_input("Home Off Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            home_def_missing = st.number_input("Home Def Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            home_l5_tt = st.number_input("Home L5 Team Total (optional)", 0.0, 200.0, 0.0, 0.5)
        with c4:
            away_off_missing = st.number_input("Away Off Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            away_def_missing = st.number_input("Away Def Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            away_l5_tt = st.number_input("Away L5 Team Total (optional)", 0.0, 200.0, 0.0, 0.5)

    submitted = st.form_submit_button("Predict Game")

if submitted:
    home_l5_tt_opt = None if home_l5_tt == 0.0 else home_l5_tt
    away_l5_tt_opt = None if away_l5_tt == 0.0 else away_l5_tt

    si = SimpleInputs(
        home_team, away_team, home_net, away_net, home_l5, away_l5,
        home_elo, away_elo, market_spread_home_minus, market_total,
        home_off_missing, home_def_missing, away_off_missing, away_def_missing,
        home_l5_team_total=home_l5_tt_opt, away_l5_team_total=away_l5_tt_opt
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

    ranked = sorted(
        [g for g in st.session_state.slate if g['output']['confidence_pct'] >= 70],
        key=lambda x: x['output']['confidence_pct'], reverse=True
    )

    st.markdown("### ⭐ Top Plays (Conf ≥ 70%)")
    if ranked:
        for i, g in enumerate(ranked[:3], 1):
            st.write(f"**{i}.** {g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']} ({g['output']['confidence_pct']}% conf)")
    else:
        st.info("No plays ≥ 70% confidence.")

    lock_candidates = []
    for g in ranked:
        meta = g['output'].get('meta', {})
        away_fav_by = meta.get('away_fav_by', 0.0)
        pick = g['output']['side_play']
        pick_is_home = pick.startswith(g['inputs']['home_team'])
        fades_road_fav = (away_fav_by >= ROAD_FAV_NOGO2) and pick_is_home
        if (g['output']['confidence_pct'] >= 75) and (not fades_road_fav):
            lock_candidates.append(g)
    if lock_candidates:
        g = lock_candidates[0]
        st.markdown("### 🔒 Lock of the Day")
        st.success(f"{g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']}  ({g['output']['confidence_pct']}% confidence)")
    else:
        st.info("No safe Lock meeting the ≥75% & no-road-fade rules.")

    if st.button("🗑️ Clear Slate"):
        st.session_state.slate = []
        st.info("Slate cleared.")
else:
    st.info("Add a game result above to build the slate and get Top 3 + 🔒 Lock.")
