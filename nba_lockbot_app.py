import math
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

"""
NBA Prediction — **Simple Version (like our NFL)**
Inputs kept minimal per game:
- Teams
- Home Net Rating, Away Net Rating (Net = ORtg − DRtg)
- Last-5 Avg Margin (Home, Away)
- Power/ELO (Home, Away)
- Pace (single average for the matchup) — default provided
- Market Spread (home negative if favored), Market Total

Outputs:
- Home/Away win %
- Predicted score & total
- Side pick (ML / ±1.5 with safer logic)
- Total pick (Over/Under/Pass)
- Confidence %
- Build a slate, Top 3, and 🔒 Lock of the Day
"""

st.set_page_config(page_title="NBA Predictor — Simple", layout="wide")
st.title("🏀 NBA Prediction — Simple (Like Our NFL)")
st.caption("Minimal inputs. Same workflow: add games, see Top 3, and one 🔒 Lock of the Day.")

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
    pace: float          # possessions per 48 (avg for game)
    market_spread_home_minus: float  # home favored = negative
    market_total: float


def predict(si: SimpleInputs) -> Dict:
    elo_diff = si.home_elo - si.away_elo
    net_diff = si.home_net - si.away_net
    form_diff = si.home_l5 - si.away_l5

    # Linear score to win prob (very compact model)
    w_hca = 1.1          # home court
    w_elo = 0.008        # per elo point
    w_net = 0.075        # per net rating point
    w_form = 0.06        # per margin point

    linear = (w_hca + w_elo*elo_diff + w_net*net_diff + w_form*form_diff)
    home_wp = logistic(linear)
    away_wp = 1 - home_wp

    # Predicted total using pace and net
    # Start from league baseline, nudge by net & form
    net_nudge = 0.35 * (si.home_net + si.away_net)
    form_nudge = 0.25 * (si.home_l5 + si.away_l5)
    total_per100 = (LEAGUE_ORtg*2) + net_nudge + form_nudge  # per 100 poss
    pred_total = clamp(si.pace * total_per100 / 100.0, 180, 270)

    # Expected margin
    exp_margin = (home_wp - 0.5) * 20 + 0.4 * net_diff + 0.25 * form_diff

    # Split total into team points by relative net strength
    share = clamp(0.5 + 0.25 * (net_diff / 20.0), 0.35, 0.65)
    home_pts = clamp(pred_total * share + 0.4*exp_margin, 70, 160)
    away_pts = clamp(pred_total - home_pts, 70, 160)

    # Picks
    # Side safer logic
    if home_wp >= 0.64 and exp_margin >= 6:
        side = f"{si.home_team} -1.5"
    elif 0.47 <= home_wp <= 0.53:
        side = f"{si.home_team} +1.5" if home_wp < 0.5 else f"{si.away_team} +1.5"
    else:
        side = f"{si.home_team} ML" if home_wp >= 0.5 else f"{si.away_team} ML"

    # Total pick
    total_edge = pred_total - si.market_total
    if total_edge >= 3.0:
        total_pick = f"Over {si.market_total:.1f}"
    elif total_edge <= -3.0:
        total_pick = f"Under {si.market_total:.1f}"
    else:
        total_pick = "Pass total"

    # Confidence (compact)
    comp = [
        min(abs(elo_diff)/50.0, 1.2),
        min(abs(net_diff)/6.0, 1.2),
        min(abs(form_diff)/6.0, 1.0),
        min(abs(total_edge)/8.0, 1.0)*0.6,
    ]
    conf = clamp(48 + 34*np.tanh(sum(comp)), 50, 86)

    return {
        "home_win_prob": round(100*home_wp, 1),
        "away_win_prob": round(100*away_wp, 1),
        "pred_home": round(home_pts, 1),
        "pred_away": round(away_pts, 1),
        "pred_total": round(pred_total, 1),
        "expected_margin": round(exp_margin, 2),
        "side_play": side,
        "total_play": total_pick,
        "confidence_pct": round(conf, 1),
    }

# ---------------------------
# UI
# ---------------------------
if 'slate' not in st.session_state:
    st.session_state.slate = []

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
        pace = st.number_input("Game Pace (poss/48)", 90.0, 110.0, 99.5, 0.1)
        market_spread_home_minus = st.number_input("Market Spread (home negative)", -20.0, 20.0, -3.5, 0.5)
        market_total = st.number_input("Market Total", 170.0, 280.0, 228.5, 0.5)

    submitted = st.form_submit_button("Predict Game")

if submitted:
    si = SimpleInputs(
        home_team, away_team, home_net, away_net, home_l5, away_l5,
        home_elo, away_elo, pace, market_spread_home_minus, market_total
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

    ranked = sorted(st.session_state.slate, key=lambda x: x['output']['confidence_pct'], reverse=True)
    st.markdown("### ⭐ Top 3 Plays")
    for i, g in enumerate(ranked[:3], 1):
        st.write(f"**{i}.** {g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']} ({g['output']['confidence_pct']}% conf)")

    lock = ranked[0] if ranked else None
    if lock:
        g = lock
        st.markdown("### 🔒 Lock of the Day")
        st.success(f"{g['inputs']['away_team']} @ {g['inputs']['home_team']} — {g['output']['side_play']} | {g['output']['total_play']}  ({g['output']['confidence_pct']}% confidence)")

    if st.button("🗑️ Clear Slate"):
        st.session_state.slate = []
        st.info("Slate cleared.")
else:
    st.info("Add a game result above to build the slate and get Top 3 + 🔒 Lock.")

st.markdown("""
**What to enter**
- **Net Rating:** ORtg − DRtg. If you only have Net, you're good — no ORtg/DRtg needed.
- **L5 Avg Margin:** Avg point diff over last 5 games.
- **Power/ELO:** Any power number you use (e.g., 1500–1800 scale). Higher = better.
- **Pace:** Possessions per 48 for the matchup. If unknown, ~99.5 is fine.
- **Market Spread:** Negative if home is favored (e.g., -4.5). Positive if home is an underdog.
- **Market Total:** Sportsbook total for the game.
""")
