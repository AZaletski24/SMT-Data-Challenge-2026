"""
SMT Data Challenge 2026 — Guess the Pitch

Simulated at-bat against a real pitcher from the dataset. Each pitch is
the pitcher's actual ~20Hz tracked ball flight. After watching the pitch:

  Phase 1 (3s enforced): identify the pitch type from the pitcher's real
  arsenal (Fastball / Slider / Curveball / etc.)

  Phase 2 (3s enforced): decide Swing or Take

At-bat outcome logic (decided with the user 2026-07-01):
  - Correct ID + Swing + in-zone  → PUT IN PLAY  (at-bat ends)
  - Any Swing + wrong ID          → STRIKE        (whiff — wrong read)
  - Correct ID + Swing + out-zone → STRIKE        (chase — bad discipline)
  - Take + in-zone                → STRIKE        (called strike)
  - Take + out-zone               → BALL
  - Timeout on pitch type         → treated as wrong ID
  - Timeout on swing/take         → treated as Take
  4 balls → Walk; 3 strikes → Strikeout; in-play → at-bat ends.

End-of-at-bat report: pitch-type identification accuracy and swing/take
decision accuracy (no external baseline — raw accuracy only).

Countdown enforcement via streamlit-autorefresh (100ms reruns + elapsed
time check against session_state timestamp).

Trajectories are pre-baked in Data/derived/pitcher_pools.json by
scripts/build_pitcher_pools.py (run locally once; JSON committed to git
so the app works on Streamlit Cloud without the raw ball-positions data).
"""
import json
import os
import time
import random

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(REPO_ROOT, "Data", "derived")

PITCHER_MIN_PITCHES = 40
COUNTDOWN_S = 3
ANIM_DURATION_MS = 1200
ZONE_HEIGHT = (1.5, 3.5)
ZONE_WIDTH = (-0.708, 0.708)


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_arsenals() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DERIVED, "pitcher_arsenals.csv"))


@st.cache_data
def load_pitcher_pools() -> dict:
    """Pre-built pitcher→pitch-pool mapping from build_pitcher_pools.py.
    Keyed by pitcher_id; each value is a list of pitch dicts with frames
    already embedded — no raw ball-positions data needed at runtime."""
    with open(os.path.join(DERIVED, "pitcher_pools.json")) as f:
        return json.load(f)


def load_eligible_pitchers() -> list[str]:
    pools = load_pitcher_pools()
    return list(pools.keys())


# ── Game logic ────────────────────────────────────────────────────────────────

def resolve_outcome(pitch: dict, id_guess: str | None, swing_take: str) -> str:
    """
    Returns: 'in_play' | 'strike' | 'ball'
    id_guess=None means timeout (treated as wrong ID).
    swing_take='take' is the timeout default for the second phase.
    """
    correct_id = (id_guess == pitch["pitch_type_label"])
    in_zone = (pitch["zone"] == "in_zone")

    if swing_take == "swing":
        if correct_id and in_zone:
            return "in_play"
        return "strike"  # wrong ID + any zone, or correct ID + out-of-zone chase
    else:  # take
        return "strike" if in_zone else "ball"


# ── Canvas HTML ───────────────────────────────────────────────────────────────

def _canvas_html(pitch: dict, mode: str, correct: bool | None = None) -> str:
    """mode: 'throw' | 'reveal'"""
    import json as _json
    payload = _json.dumps({
        "frames": pitch["frames"],
        "plate_x": pitch["plate_x"],
        "plate_z": pitch["plate_z"],
        "zone": pitch["zone"],
        "mode": mode,
        "correct": correct,
        "zone_height": list(ZONE_HEIGHT),
        "zone_width": list(ZONE_WIDTH),
        "anim_ms": ANIM_DURATION_MS,
    })
    return f"""
<canvas id="pitch-canvas" width="380" height="460"
  style="display:block;margin:0 auto;border-radius:8px"></canvas>
<script>
const DATA = {payload};
const canvas = document.getElementById('pitch-canvas');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const CAM = {{x:0, y:-3.5, z:3.5}};
const F = 160, CX = W/2, CY = 160, HORIZON = CY, DIRT_Y = CY + 70;

function project(x3,y3,z3) {{
  const dy = y3 - CAM.y;
  if (dy <= 0.1) return null;
  return {{sx: CX + F*(x3-CAM.x)/dy, sy: CY - F*(z3-CAM.z)/dy,
           r: 2 + Math.min(13, F*0.09/dy)}};
}}
function drawField() {{
  ctx.clearRect(0,0,W,H);
  const sky = ctx.createLinearGradient(0,0,0,HORIZON);
  sky.addColorStop(0,'#07111e'); sky.addColorStop(0.45,'#0f2d52'); sky.addColorStop(1,'#1e5490');
  ctx.fillStyle = sky; ctx.fillRect(0,0,W,HORIZON);
  const grass = ctx.createLinearGradient(0,HORIZON,0,DIRT_Y);
  grass.addColorStop(0,'#1b4a17'); grass.addColorStop(1,'#2c7028');
  ctx.fillStyle = grass; ctx.fillRect(0,HORIZON,W,DIRT_Y-HORIZON);
  const dirt = ctx.createLinearGradient(0,DIRT_Y,0,H);
  dirt.addColorStop(0,'#7a4d2f'); dirt.addColorStop(0.5,'#6b4228'); dirt.addColorStop(1,'#563318');
  ctx.fillStyle = dirt; ctx.fillRect(0,DIRT_Y,W,H-DIRT_Y);
  ctx.lineWidth=1.2; ctx.strokeStyle='rgba(240,240,220,0.18)';
  ctx.beginPath(); ctx.moveTo(0,H); ctx.lineTo(CX,HORIZON); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W,H); ctx.lineTo(CX,HORIZON); ctx.stroke();
  const platePts = [[-0.708,0.08,0],[0.708,0.08,0],[0.708,-0.42,0],
    [0,-0.71,0],[-0.708,-0.42,0]].map(([x,y,z])=>project(x,y,z)).filter(Boolean);
  if (platePts.length===5) {{
    ctx.fillStyle='#f2f2f2'; ctx.strokeStyle='#bbb'; ctx.lineWidth=1.5;
    ctx.beginPath();
    platePts.forEach((p,i)=>i===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy));
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }}
  ctx.strokeStyle='rgba(245,245,230,0.65)'; ctx.lineWidth=1.8;
  [[[-1.7,-0.75,0],[-0.708,-0.75,0],[-0.708,0.5,0],[-1.7,0.5,0]],
   [[0.708,-0.75,0],[1.7,-0.75,0],[1.7,0.5,0],[0.708,0.5,0]]
  ].forEach(corners=>{{
    const pts=corners.map(([x,y,z])=>project(x,y,z)).filter(Boolean);
    if (pts.length===4) {{
      ctx.beginPath();
      pts.forEach((p,i)=>i===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy));
      ctx.closePath(); ctx.stroke();
    }}
  }});
}}
function drawZoneAndLanding() {{
  const [zb,zt]=DATA.zone_height, [zl_x,zr_x]=DATA.zone_width;
  const zl=project(zl_x,0,zt), zr=project(zr_x,0,zt), zbm=project(zl_x,0,zb);
  if (zl&&zr&&zbm) {{
    const zx=zl.sx, zy=zl.sy, zw=zr.sx-zl.sx, zh=zbm.sy-zl.sy;
    ctx.fillStyle='rgba(233,196,106,0.07)'; ctx.fillRect(zx,zy,zw,zh);
    ctx.strokeStyle='#e9c46a'; ctx.lineWidth=2.5; ctx.setLineDash([6,4]);
    ctx.strokeRect(zx,zy,zw,zh); ctx.setLineDash([]);
    ctx.fillStyle='#e9c46a'; ctx.font='bold 10px monospace';
    ctx.textAlign='center'; ctx.fillText('STRIKE ZONE',zx+zw/2,zy-8);
  }}
  const pt=project(DATA.plate_x,0,DATA.plate_z);
  if (pt) {{
    const color=DATA.correct===null?'#fff':(DATA.correct?'#2dd36f':'#e5484d');
    ctx.beginPath(); ctx.arc(pt.sx,pt.sy,10,0,Math.PI*2);
    ctx.fillStyle=color; ctx.globalAlpha=0.9; ctx.fill(); ctx.globalAlpha=1;
    ctx.strokeStyle='#222'; ctx.lineWidth=1.5; ctx.stroke();
  }}
}}
function drawBall(sx,sy,r) {{
  ctx.beginPath(); ctx.arc(sx,sy,r+3,0,Math.PI*2);
  ctx.fillStyle='rgba(255,255,255,0.08)'; ctx.fill();
  const grad=ctx.createRadialGradient(sx-r*.35,sy-r*.35,r*.08,sx,sy,r);
  grad.addColorStop(0,'#ffffff'); grad.addColorStop(0.55,'#e8e8e8'); grad.addColorStop(1,'#b8b8b8');
  ctx.beginPath(); ctx.arc(sx,sy,r,0,Math.PI*2);
  ctx.fillStyle=grad; ctx.fill();
  ctx.strokeStyle='rgba(100,100,100,0.5)'; ctx.lineWidth=Math.max(0.4,r*0.1); ctx.stroke();
}}
function frameAt(frac) {{
  const frames=DATA.frames, pos=frac*(frames.length-1);
  const i0=Math.floor(pos), i1=Math.min(i0+1,frames.length-1), t=pos-i0;
  const a=frames[i0], b=frames[i1];
  return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
}}
if (DATA.mode==='throw') {{
  let start=null;
  function tick(ts) {{
    if (!start) start=ts;
    const frac=Math.min((ts-start)/DATA.anim_ms,1);
    drawField();
    const [x,y,z]=frameAt(frac);
    const pt=project(x,y,z);
    if (pt) drawBall(pt.sx,pt.sy,pt.r);
    if (frac<1) requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}} else {{
  drawField(); drawZoneAndLanding();
}}
</script>"""


# ── Session state helpers ─────────────────────────────────────────────────────

def _next_pitch():
    """Advance to the next pitch in the at-bat pool."""
    st.session_state.pitch_idx += 1
    st.session_state.phase = "throw"
    st.session_state.id_guess = None
    st.session_state.swing_take = None


def _record(pitch: dict, id_guess: str | None, swing_take: str):
    outcome = resolve_outcome(pitch, id_guess, swing_take)
    correct_id = (id_guess == pitch["pitch_type_label"])

    if outcome == "ball":
        st.session_state.balls += 1
    elif outcome == "strike":
        st.session_state.strikes += 1

    st.session_state.history.append({
        "pitch_type_label": pitch["pitch_type_label"],
        "zone": pitch["zone"],
        "release_speed_mph": pitch["release_speed_mph"],
        "id_guess": id_guess,
        "correct_id": correct_id,
        "swing_take": swing_take,
        "outcome": outcome,
    })

    at_bat_over = (
        st.session_state.balls >= 4 or
        st.session_state.strikes >= 3 or
        outcome == "in_play"
    )

    st.session_state.last_outcome = outcome
    st.session_state.last_correct_id = correct_id
    st.session_state.phase = "at_bat_over" if at_bat_over else "revealed"
    st.rerun()


def init_game(pitcher_id: str):
    pool = list(load_pitcher_pools()[pitcher_id])

    if len(pool) < 6:
        st.error("Not enough pitchable data for this pitcher. Pick another.")
        return

    random.shuffle(pool)
    st.session_state.pitcher_id = pitcher_id
    st.session_state.pitch_pool = pool
    st.session_state.pitch_idx = 0
    st.session_state.phase = "throw"
    st.session_state.balls = 0
    st.session_state.strikes = 0
    st.session_state.history = []
    st.session_state.id_guess = None
    st.session_state.swing_take = None
    st.session_state.last_outcome = None
    st.session_state.last_correct_id = None
    st.rerun()


# ── Count display ─────────────────────────────────────────────────────────────

def _show_count():
    b = st.session_state.balls
    s = st.session_state.strikes
    balls_disp = "●" * b + "○" * (3 - b)
    strikes_disp = "●" * s + "○" * (2 - s)
    st.caption(f"Count — Balls: {balls_disp}  Strikes: {strikes_disp}")


# ── Phase renderers ───────────────────────────────────────────────────────────

def render_start():
    st.title("Guess the Pitch")
    st.markdown(
        "Step in against a real pitcher from the SMT tracking dataset. "
        "Watch each pitch travel to the plate, then identify the pitch type "
        "and decide whether to Swing or Take — you have **3 seconds** for each. "
        "Work the count: earn a walk with 4 correct takes, put the ball in play "
        "by correctly reading a strike and swinging, or strike out after 3 misses."
    )

    eligible = load_eligible_pitchers()
    pitcher_id = random.choice(eligible)

    arsenal = load_arsenals()
    pitcher_arsenal = arsenal[arsenal["pitcher"] == pitcher_id].sort_values(
        "usage_pct", ascending=False
    )

    st.subheader(f"Today's pitcher: `{pitcher_id}`")
    st.markdown(f"**{pitcher_arsenal['total_pitches'].iloc[0]} pitches tracked this season.**")

    rows = []
    for _, r in pitcher_arsenal.iterrows():
        rows.append({
            "Pitch": r["pitch_type_label"],
            "Usage": f"{r['usage_pct']*100:.0f}%",
            "Avg Velo": f"{r['avg_release_speed_mph']:.1f} mph",
            "Avg IVB": f"{r['avg_ivb_in']:+.1f} in",
            "Avg HB": f"{r['avg_hb_in']:+.1f} in",
        })
    st.table(rows)

    col1, col2 = st.columns(2)
    if col1.button("⚾ Face This Pitcher", type="primary", use_container_width=True):
        init_game(pitcher_id)
    if col2.button("🔀 Pick Different Pitcher", use_container_width=True):
        st.rerun()


def render_throw():
    pitch = st.session_state.pitch_pool[st.session_state.pitch_idx]
    n = len(st.session_state.history) + 1
    st.subheader(f"Pitch {n}")
    _show_count()
    components.html(_canvas_html(pitch, "throw"), height=470)
    if st.button("What pitch was that? →", type="primary"):
        st.session_state.phase = "pitch_type"
        st.session_state.phase_start = time.time()
        st.rerun()


def render_pitch_type():
    arsenal = load_arsenals()
    pitch_types = (
        arsenal[arsenal["pitcher"] == st.session_state.pitcher_id]
        .sort_values("usage_pct", ascending=False)["pitch_type_label"]
        .tolist()
    )

    elapsed = time.time() - st.session_state.phase_start
    remaining = max(0.0, COUNTDOWN_S - elapsed)

    if remaining <= 0:
        st.session_state.id_guess = None  # timeout = wrong ID
        st.session_state.phase = "swing_take"
        st.session_state.phase_start = time.time()
        st.rerun()
        return

    st_autorefresh(interval=100, key="pt_refresh")

    st.subheader("What pitch was that?")
    _show_count()
    st.progress(remaining / COUNTDOWN_S, text=f"⏱ {remaining:.1f}s")

    cols = st.columns(min(len(pitch_types), 3))
    for i, pt in enumerate(pitch_types):
        if cols[i % 3].button(pt, key=f"pt_{pt}", use_container_width=True):
            st.session_state.id_guess = pt
            st.session_state.phase = "swing_take"
            st.session_state.phase_start = time.time()
            st.rerun()


def render_swing_take():
    elapsed = time.time() - st.session_state.phase_start
    remaining = max(0.0, COUNTDOWN_S - elapsed)
    pitch = st.session_state.pitch_pool[st.session_state.pitch_idx]

    if remaining <= 0:
        _record(pitch, st.session_state.id_guess, "take")  # timeout = Take
        return

    st_autorefresh(interval=100, key="st_refresh")

    id_guess = st.session_state.id_guess
    guess_label = id_guess if id_guess else "*(timed out)*"
    st.subheader(f"Swing or Take? — you called **{guess_label}**")
    _show_count()
    st.progress(remaining / COUNTDOWN_S, text=f"⏱ {remaining:.1f}s")

    col1, col2 = st.columns(2)
    if col1.button("🪃 Swing", type="primary", use_container_width=True):
        _record(pitch, id_guess, "swing")
    if col2.button("🤚 Take", use_container_width=True):
        _record(pitch, id_guess, "take")


def render_revealed():
    pitch = st.session_state.pitch_pool[st.session_state.pitch_idx]
    outcome = st.session_state.last_outcome
    correct_id = st.session_state.last_correct_id
    id_guess = st.session_state.history[-1]["id_guess"]
    swing_take = st.session_state.history[-1]["swing_take"]

    correct_decision = (outcome == "ball" or outcome == "in_play")
    components.html(_canvas_html(pitch, "reveal", correct_decision), height=470)

    zone_label = "in the zone" if pitch["zone"] == "in_zone" else "out of the zone"
    id_result = f"✅ **{pitch['pitch_type_label']}** — correct!" if correct_id \
        else f"❌ It was a **{pitch['pitch_type_label']}**, you guessed **{id_guess or 'nothing (timed out)'}**"
    st.markdown(id_result)

    if outcome == "ball":
        st.success(f"Ball — good take on a pitch {zone_label}.")
    elif outcome == "strike" and swing_take == "take":
        st.error(f"Strike — called. That pitch was {zone_label}.")
    elif outcome == "strike":
        st.error(f"Strike — swinging miss.")
    _show_count()

    if st.button("Next pitch →", type="primary"):
        _next_pitch()
        st.rerun()


def render_at_bat_over():
    pitch = st.session_state.pitch_pool[st.session_state.pitch_idx]
    outcome = st.session_state.last_outcome
    correct_id = st.session_state.last_correct_id
    id_guess = st.session_state.history[-1]["id_guess"]

    # Show final-pitch reveal
    correct_decision = (outcome == "in_play")
    components.html(_canvas_html(pitch, "reveal", correct_decision), height=470)

    id_result = f"✅ **{pitch['pitch_type_label']}**" if correct_id \
        else f"❌ **{pitch['pitch_type_label']}** (you guessed **{id_guess or 'nothing'}**)"
    st.markdown(id_result)

    balls = st.session_state.balls
    strikes = st.session_state.strikes
    if balls >= 4:
        st.success("## ⚾ Walk! You worked the count to 4 balls.")
    elif strikes >= 3:
        st.error("## 🔴 Strikeout.")
    else:
        st.info("## 🏃 Ball in Play!")

    if st.button("See Results", type="primary"):
        st.session_state.phase = "results"
        st.rerun()


def render_results():
    history = st.session_state.history
    pitcher_id = st.session_state.pitcher_id
    balls = st.session_state.balls
    strikes = st.session_state.strikes

    if balls >= 4:
        outcome_str = "Walk"
        outcome_icon = "⚾"
    elif strikes >= 3:
        outcome_str = "Strikeout"
        outcome_icon = "🔴"
    else:
        outcome_str = "Ball in Play"
        outcome_icon = "🏃"

    st.title(f"{outcome_icon} {outcome_str}")
    st.caption(f"vs. `{pitcher_id}` — {len(history)} pitches seen")

    n = len(history)
    id_correct = sum(1 for h in history if h["correct_id"])
    st_correct = sum(1 for h in history if h["outcome"] in ("ball", "in_play"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Pitch ID Accuracy", f"{id_correct}/{n}",
                f"{id_correct/n*100:.0f}%")
    col2.metric("Swing/Take Accuracy", f"{st_correct}/{n}",
                f"{st_correct/n*100:.0f}%")
    col3.metric("Final Count", f"{balls}B - {strikes}S")

    rows = []
    for i, h in enumerate(history):
        id_mark = "✅" if h["correct_id"] else "❌"
        st_mark = "✅" if h["outcome"] in ("ball", "in_play") else "❌"
        rows.append({
            "#": i + 1,
            "Actual Pitch": h["pitch_type_label"],
            "Your Guess": h["id_guess"] or "timeout",
            "ID": id_mark,
            "Velo": f"{h['release_speed_mph']:.0f} mph",
            "Zone": "Strike" if h["zone"] == "in_zone" else "Ball",
            "Decision": h["swing_take"] or "timeout",
            "S/T": st_mark,
            "Outcome": h["outcome"].replace("_", " ").title(),
        })
    st.table(rows)

    if st.button("Play Again", type="primary"):
        for key in ["pitcher_id", "pitch_pool", "pitch_idx", "phase",
                    "balls", "strikes", "history", "id_guess", "swing_take",
                    "last_outcome", "last_correct_id", "phase_start"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Guess the Pitch", page_icon="⚾", layout="centered")

    phase = st.session_state.get("phase")

    if phase is None:
        render_start()
    elif phase == "throw":
        render_throw()
    elif phase == "pitch_type":
        render_pitch_type()
    elif phase == "swing_take":
        render_swing_take()
    elif phase == "revealed":
        render_revealed()
    elif phase == "at_bat_over":
        render_at_bat_over()
    elif phase == "results":
        render_results()


if __name__ == "__main__":
    main()
