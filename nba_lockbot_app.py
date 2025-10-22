import math
from dataclasses import dataclass, asdict
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# NBA Prediction Model (Manual-Input) — Like our NFL version
# -------------------------------------------------------------
# Notes
# - No external APIs: you can paste numbers from your research.
# - Produces: Win % (home & away), predicted score, ML/spread/total pick,
#   confidence %, and marks a single 🔒 Lock of the Day for a slate.
# - Safer-pick logic: default to ML; only use -spread for dominant edges,
#   and +spread for live dogs in tight games.
# -------------------------------------------------------------

st.set_page_config(page_title="NBA Predictor — Like NFL", layout="wide")
st.title("🏀 NBA Prediction — Like Our NFL Model")
st.caption("Manual-input model with win %, predicted score, recommended play, and confidence. Add multiple games to build a slate, Top 3, and one 🔒 Lock of the Day.")

# ---------------------------
# Helpers & core math
# ---------------------------
@dataclass
class GameInputs:
    home_team: str
    away_team: str

    # Ratings (per 100 possessions)
    home_off: float  # ORtg
    home_def: float  # DRtg
    away_off: float
    away_def: float

    # Recent form (last 5 games)
    home_l5_margin: float  # avg point diff last 5
    away_l5_margin: float

    # Pace (possessions per 48)
    home_pace: float
    away_pace: float

    # ELO or power ratings
    home_elo: float
    away_elo: float

    # Injuries/availability (estimate missing points per 100)
    home_off_missing: float
    away_off_missing: float
    home_def_missing: float
    away_def_missing: float

    # Rest & schedule
    home_days_rest: float
    away_days_rest: float
    home_b2b: bool
    away_b2b: bool
    travel_miles_home: float
    travel_miles_away: float

    # Market lines (optional, for bet reco)
    market_spread_home_minus: float  # negative means home favored; ex: -4.5
    market_total: float

    # Trends (optional, small weights)
    home_ou_trend_delta: float  # avg delta from closing total last 5 ("+Over")
    away_ou_trend_delta: float


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def model_predict(inputs: GameInputs) -> Dict:
    # --- Feature engineering ---
    home_net = inputs.home_off - inputs.home_def
    away_net = inputs.away_off - inputs.away_def
    net_diff = home_net - away_net

    elo_diff = inputs.home_elo - inputs.away_elo
    form_diff = inputs.home_l5_margin - inputs.away_l5_margin

    # Adjust for injuries (offense reduced by off_missing, defense worsened by def_missing)
    adj_home_off = inputs.home_off - inputs.home_off_missing
    adj_home_def = inputs.home_def + inputs.home_def_missing
    adj_away_off = inputs.away_off - inputs.away_off_missing
    adj_away_def = inputs.away_def + inputs.away_def_missing

    adj_home_net = adj_home_off - adj_home_def
    adj_away_net = adj_away_off - adj_away_def
    adj_net_diff = adj_home_net - adj_away_net

    # Rest & travel edges
    rest_edge = (inputs.home_days_rest - inputs.away_days_rest)
    b2b_edge = (0.0
                + (0.8 if inputs.away_b2b else 0.0)
                - (0.8 if inputs.home_b2b else 0.0))
    travel_edge = clamp((inputs.away_travel_miles - inputs.home_travel_miles) / 1000.0, -2, 2)

    # Pace estimate (blend)
    pace = 0.5 * (inputs.home_pace + inputs.away_pace)

    # --- Scoring function (linear -> logistic) ---
    # Calibrated weights (tweakable)
    w = {
        'base_hca': 1.2,         # baseline home-court edge
        'elo': 0.008,            # per ELO point
        'net': 0.07,             # per net rating point
        'adj_net': 0.06,         # injuries-adjusted net diff
        'form': 0.06,            # last 5 margin
        'rest': 0.12,            # per day of rest edge
        'b2b': 0.45,             # back-to-back edge units
        'travel': 0.18,          # per 1000 miles rel.
    }

    linear_score = (
        w['base_hca']
        + w['elo'] * elo_diff
        + w['net'] * net_diff
        + w['adj_net'] * adj_net_diff
        + w['form'] * form_diff
        + w['rest'] * rest_edge
        + w['b2b'] * b2b_edge
        + w['travel'] * travel_edge
    )

    home_win_prob = logistic(linear_score)
    away_win_prob = 1 - home_win_prob

    # --- Predict points ---
    # Offensive and defensive matchup adjustment
    home_off_vs_opp = adj_home_off - 0.5 * (adj_away_def - 112)
    away_off_vs_opp = adj_away_off - 0.5 * (adj_home_def - 112)

    # Convert to per-game using pace: total = pace * (off1 + off2)/100
    base_total = pace * (home_off_vs_opp + away_off_vs_opp) / 100.0

    # Add OU trend nudges (small)
    total_nudge = 0.15 * (inputs.home_ou_trend_delta + inputs.away_ou_trend_delta)
    predicted_total = clamp(base_total + total_nudge, 180, 270)

    # Expected margin from win prob via inverse logistic slope (empirical scale)
    expected_margin = (home_win_prob - 0.5) * 20 + 0.35 * (adj_net_diff)

    # Split total into team scores by offensive share
    share = clamp((home_off_vs_opp) / (home_off_vs_opp + away_off_vs_opp + 1e-6), 0.3, 0.7)
    home_pts = clamp(predicted_total * share + 0.4 * expected_margin, 70, 160)
    away_pts = clamp(predicted_total - home_pts, 70, 160)

    # --- Bet Recommendation Logic ---
    # Safer defaults: ML by default; -spread only if strong edge; +spread for live dogs in tight games.
    spread_edge = expected_margin - (-inputs.market_spread_home_minus)  # how much we beat the book's line

    if home_win_prob >= 0.63 and expected_margin >= 6:
        play = f"{inputs.home_team} -1.5"
    elif 0.48 <= home_win_prob <= 0.55:
        # tight game — consider dog +1.5
        if home_win_prob < 0.5:
            play = f"{inputs.home_team} +1.5"
        else:
            play = f"{inputs.away_team} +1.5"
    else:
        # ML default on the side with higher win prob
        play = f"{inputs.home_team} ML" if home_win_prob >= 0.5 else f"{inputs.away_team} ML"

    # Total pick
    total_edge = predicted_total - inputs.market_total
    if total_edge >= 3.0:
        total_pick = f"Over {inputs.market_total:.1f}"
    elif total_edge <= -3.0:
        total_pick = f"Under {inputs.market_total:.1f}"
    else:
        total_pick = "Pass total"

    # Confidence: combine signal strengths (normalized 0-100)
    components = [
        min(abs(elo_diff) / 50.0, 1.2),
        min(abs(adj_net_diff) / 6.0, 1.2),
        min(abs(form_diff) / 6.0, 1.0),
        min(abs(rest_edge) / 2.0, 1.0),
        min(abs(total_edge) / 8.0, 1.0) * 0.6,
    ]
    conf = clamp(45 + 35 * np.tanh(sum(components)), 50, 85)

    return {
        'home_win_prob': round(100 * home_win_prob, 1),
        'away_win_prob': round(100 * away_win_prob, 1),
        'pred_home': round(home_pts, 1),
        'pred_away': round(away_pts, 1),
        'pred_total': round(predicted_total, 1),
        'expected_margin': round(expected_margin, 2),
        'side_play': play,
        'total_play': total_pick,
        'confidence_pct': round(conf, 1),
    }


# ---------------------------
# UI: single game form
# ---------------------------
def single_game_form(key_prefix: str = "") -> GameInputs | None:
    with st.form(f"game_form_{key_prefix}"):
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            home_team = st.text_input("Home Team", value="Lakers")
            away_team = st.text_input("Away Team", value="Warriors")
            home_off = st.number_input("Home ORtg", 95.0, 125.0, 118.0, 0.1)
            home_def = st.number_input("Home DRtg", 95.0, 125.0, 113.0, 0.1)
            away_off = st.number_input("Away ORtg", 95.0, 125.0, 116.0, 0.1)
            away_def = st.number_input("Away DRtg", 95.0, 125.0, 114.0, 0.1)
            home_l5_margin = st.number_input("Home L5 avg margin", -20.0, 20.0, 3.0, 0.1)
            away_l5_margin = st.number_input("Away L5 avg margin", -20.0, 20.0, 1.0, 0.1)
        with c2:
            home_pace = st.number_input("Home Pace (poss/48)", 90.0, 110.0, 99.5, 0.1)
            away_pace = st.number_input("Away Pace (poss/48)", 90.0, 110.0, 100.5, 0.1)
            home_elo = st.number_input("Home ELO/Power", 1300.0, 1900.0, 1650.0, 1.0)
            away_elo = st.number_input("Away ELO/Power", 1300.0, 1900.0, 1635.0, 1.0)
            home_off_missing = st.number_input("Home Off Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            away_off_missing = st.number_input("Away Off Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            home_def_missing = st.number_input("Home Def Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
            away_def_missing = st.number_input("Away Def Missing (per 100)", 0.0, 20.0, 0.0, 0.1)
        with c3:
            home_days_rest = st.number_input("Home Days Rest", 0.0, 7.0, 2.0, 0.5)
            away_days_rest = st.number_input("Away Days Rest", 0.0, 7.0, 1.0, 0.5)
            home_b2b = st.checkbox("Home on B2B", value=False)
            away_b2b = st.checkbox("Away on B2B", value=False)
            travel_miles_home = st.number_input("Home Travel Miles", 0.0, 6000.0, 0.0, 10.0)
            travel_miles_away = st.number_input("Away Travel Miles", 0.0, 6000.0, 1200.0, 10.0)

        c4, c5 = st.columns(2)
        with c4:
            market_spread_home_minus = st.number_input("Market Spread (home negative, e.g., -4.5)", -20.0, 20.0, -3.5, 0.5)
            market_total = st.number_input("Market Total", 170.0, 280.0, 232.5, 0.5)
        with c5:
            home_ou_trend_delta = st.number_input("Home O/U trend delta (+'Over')", -12.0, 12.0, 0.5, 0.1)
            away_ou_trend_delta = st.number_input("Away O/U trend delta (+'Over')", -12.0, 12.0, -0.2, 0.1)

        submitted = st.form_submit_button("Predict Game")

    if not submitted:
        return None

    return GameInputs(
        home_team=home_team,
        away_team=away_team,
        home_off=home_off,
        home_def=home_def,
        away_off=away_off,
        away_def=away_def,
        home_l5_margin=home_l5_margin,
        away_l5_margin=away_l5_margin,
        home_pace=home_pace,
        away_pace=away_pace,
        home_elo=home_elo,
        away_elo=away_elo,
        home_off_missing=home_off_missing,
        away_off_missing=away_off_missing,
        home_def_missing=home_def_missing,
        away_def_missing=away_def_missing,
        home_days_rest=home_days_rest,
        away_days_rest=away_days_rest,
        home_b2b=home_b2b,
        away_b2b=away_b2b,
        travel_miles_home=travel_miles_home,
        travel_miles_away=travel_miles_away,
        market_spread_home_minus=market_spread_home_minus,
        market_total=market_total,
        home_ou_trend_delta=home_ou_trend_delta,
        away_ou_trend_delta=away_ou_trend_delta,
    )


# ---------------------------
# Slate management
# ---------------------------
if 'slate' not in st.session_state:
    st.session_state.slate = []  # list of dicts {inputs: GameInputs, output: dict}

left, right = st.columns([1.2, 1])
with left:
    gi = single_game_form("main")
    if gi:
        out = model_predict(gi)
        st.subheader("Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{gi.home_team} Win %", f"{out['home_win_prob']}%")
        c2.metric(f"{gi.away_team} Win %", f"{out['away_win_prob']}%")
        c3.metric("Predicted Total", out['pred_total'])
        c4.metric("Confidence", f"{out['confidence_pct']}%")

        st.write(
            f"**Predicted Score:** {gi.away_team} {out['pred_away']} @ {gi.home_team} {out['pred_home']}  ")
        st.write(f"**Side:** {out['side_play']}  |  **Total:** {out['total_play']}  |  **Edge (spread est):** {out['expected_margin']} pts")

        add = st.button("➕ Add to Slate")
        if add:
            st.session_state.slate.append({
                'inputs': asdict(gi),
                'output': out
            })
            st.success("Added to slate.")

with right:
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

        # Rank by confidence and compute Top 3 & Lock of the Day (highest conf, avoid volatile totals only)
        ranked = sorted(st.session_state.slate, key=lambda x: x['output']['confidence_pct'], reverse=True)
        st.markdown("---")
        st.markdown("### ⭐ Top 3 Plays")
        for i, g in enumerate(ranked[:3], 1):
            st.write(f"**{i}.** {g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']} ({g['output']['confidence_pct']}% conf)")

        lock = ranked[0] if ranked else None
        if lock:
            st.markdown("### 🔒 Lock of the Day")
            g = lock
            st.success(f"{g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']}  ({g['output']['confidence_pct']}% confidence)")

        if st.button("🗑️ Clear Slate"):
            st.session_state.slate = []
            st.info("Slate cleared.")
    else:
        st.info("Add games from the left panel to build a slate and get Top 3 + 🔒 Lock.")

st.markdown("""
**Tips for inputs**
- **ORtg/DRtg**: team points scored/allowed per 100 possessions (cleaningtheglass/Basketball-Reference style). If you only have Net Rating, set ORtg≈112+Net/2, DRtg≈112−Net/2.
- **L5 margin**: average point differential last 5 games.
- **Off/Def Missing**: your estimate of how much offense/defense (per 100) is lost due to injuries/minutes restrictions.
- **OU trend delta**: average of (game total − closing total) over last few games; positive leans Over.
- **Market Spread**: input as negative when home is favored (e.g., -4.5), positive if home is an underdog.
""")
