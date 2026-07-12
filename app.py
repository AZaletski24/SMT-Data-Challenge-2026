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

PITCH_COLORS = {
    "Fastball":  "#ef4444",
    "Sinker":    "#f97316",
    "Cutter":    "#a855f7",
    "Slider":    "#3b82f6",
    "Curveball": "#06b6d4",
    "Changeup":  "#22c55e",
}


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


def _arsenal_canvas_html(pool: list) -> str:
    """Static canvas showing one representative trajectory per pitch type,
    colour-coded by PITCH_COLORS. The representative is the pitch whose
    plate landing (plate_x, plate_z) is closest to that type's centroid —
    i.e. the most average example of what that pitch looks like."""
    import json as _j
    import numpy as _np

    def _representative(pitches: list) -> dict:
        pts = _np.array([[p["plate_x"], p["plate_z"]] for p in pitches])
        centroid = pts.mean(axis=0)
        idx = int(_np.linalg.norm(pts - centroid, axis=1).argmin())
        return pitches[idx]

    arsenal_data = {}
    for pt, color in PITCH_COLORS.items():
        bucket = [p for p in pool if p["pitch_type_label"] == pt]
        if bucket:
            rep = _representative(bucket)
            arsenal_data[pt] = {"color": color, "pitches": [rep["frames"]]}

    payload = _j.dumps({
        "zone_height": list(ZONE_HEIGHT),
        "zone_width":  list(ZONE_WIDTH),
        "arsenal":     arsenal_data,
    })
    return f"""
<canvas id="ac" width="380" height="460"
  style="display:block;margin:0 auto;border-radius:8px"></canvas>
<script>
const D={payload};
const cv=document.getElementById('ac'),ctx=cv.getContext('2d');
const W=cv.width,H=cv.height;
const CAM={{x:0,y:-3.5,z:3.5}};
const F=160,CX=W/2,CY=160,HZ=CY,DY=CY+70;
function proj(x,y,z){{
  const dy=y-CAM.y; if(dy<=0.1) return null;
  return {{sx:CX+F*(x-CAM.x)/dy, sy:CY-F*(z-CAM.z)/dy}};
}}
function field(){{
  ctx.clearRect(0,0,W,H);
  let g=ctx.createLinearGradient(0,0,0,HZ);
  g.addColorStop(0,'#07111e');g.addColorStop(0.45,'#0f2d52');g.addColorStop(1,'#1e5490');
  ctx.fillStyle=g;ctx.fillRect(0,0,W,HZ);
  g=ctx.createLinearGradient(0,HZ,0,DY);
  g.addColorStop(0,'#1b4a17');g.addColorStop(1,'#2c7028');
  ctx.fillStyle=g;ctx.fillRect(0,HZ,W,DY-HZ);
  g=ctx.createLinearGradient(0,DY,0,H);
  g.addColorStop(0,'#7a4d2f');g.addColorStop(0.5,'#6b4228');g.addColorStop(1,'#563318');
  ctx.fillStyle=g;ctx.fillRect(0,DY,W,H-DY);
  ctx.lineWidth=1.2;ctx.strokeStyle='rgba(240,240,220,0.18)';
  ctx.beginPath();ctx.moveTo(0,H);ctx.lineTo(CX,HZ);ctx.stroke();
  ctx.beginPath();ctx.moveTo(W,H);ctx.lineTo(CX,HZ);ctx.stroke();
  const pp=[[-0.708,0.08,0],[0.708,0.08,0],[0.708,-0.42,0],[0,-0.71,0],[-0.708,-0.42,0]]
    .map(([x,y,z])=>proj(x,y,z)).filter(Boolean);
  if(pp.length===5){{ctx.fillStyle='#f2f2f2';ctx.strokeStyle='#bbb';ctx.lineWidth=1.5;
    ctx.beginPath();pp.forEach((p,i)=>i===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy));
    ctx.closePath();ctx.fill();ctx.stroke();}}
  ctx.strokeStyle='rgba(245,245,230,0.65)';ctx.lineWidth=1.8;
  [[[-1.7,-0.75,0],[-0.708,-0.75,0],[-0.708,0.5,0],[-1.7,0.5,0]],
   [[0.708,-0.75,0],[1.7,-0.75,0],[1.7,0.5,0],[0.708,0.5,0]]
  ].forEach(cs=>{{const pts=cs.map(([x,y,z])=>proj(x,y,z)).filter(Boolean);
    if(pts.length===4){{ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy));ctx.closePath();ctx.stroke();}}}});
}}
function zone(){{
  const[zb,zt]=D.zone_height,[zl,zr]=D.zone_width;
  const tl=proj(zl,0,zt),tr=proj(zr,0,zt),bl=proj(zl,0,zb);
  if(tl&&tr&&bl){{
    const zx=tl.sx,zy=tl.sy,zw=tr.sx-tl.sx,zh=bl.sy-tl.sy;
    ctx.fillStyle='rgba(233,196,106,0.07)';ctx.fillRect(zx,zy,zw,zh);
    ctx.strokeStyle='#e9c46a';ctx.lineWidth=2.5;ctx.setLineDash([6,4]);
    ctx.strokeRect(zx,zy,zw,zh);ctx.setLineDash([]);
    ctx.fillStyle='#e9c46a';ctx.font='bold 10px monospace';
    ctx.textAlign='center';ctx.fillText('STRIKE ZONE',zx+zw/2,zy-8);
  }}
}}
field(); zone();
for(const[type,data]of Object.entries(D.arsenal)){{
  data.pitches.forEach(frames=>{{
    const pts=frames.map(([x,y,z])=>proj(x,y,z)).filter(Boolean);
    if(pts.length<2)return;
    ctx.strokeStyle=data.color+'55';ctx.lineWidth=1.2;
    ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy));
    ctx.stroke();
    const ep=pts[pts.length-1];
    ctx.beginPath();ctx.arc(ep.sx,ep.sy,3.5,0,Math.PI*2);
    ctx.fillStyle=data.color;ctx.fill();
  }});
}}
let lx=12,ly=20;ctx.font='bold 11px monospace';ctx.textBaseline='middle';
for(const[type,data]of Object.entries(D.arsenal)){{
  ctx.fillStyle=data.color;
  ctx.beginPath();ctx.arc(lx+5,ly,5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='rgba(255,255,255,0.9)';ctx.textAlign='left';
  ctx.fillText(type,lx+14,ly);ly+=18;
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

def get_demo_content() -> tuple[dict, list[str]]:
    """Pick a demo pitch and pitch-type button list for the tutorial.
    Prefers a pitcher with multiple types and an in-zone pitch."""
    pools    = load_pitcher_pools()
    arsenals = load_arsenals()
    n_types  = arsenals.groupby("pitcher").size().sort_values(ascending=False)
    for pid in n_types.index:
        if pid not in pools:
            continue
        in_zone = [p for p in pools[pid] if p["zone"] == "in_zone"]
        if in_zone:
            pitch      = in_zone[0]
            pitch_types = (
                arsenals[arsenals["pitcher"] == pid]
                .sort_values("usage_pct", ascending=False)["pitch_type_label"]
                .tolist()
            )
            return pitch, pitch_types
    # Fallback
    pid = list(pools.keys())[0]
    return pools[pid][0], arsenals[arsenals["pitcher"] == pid]["pitch_type_label"].tolist()


def render_demo():
    """Auto-advancing tutorial. Only user input: clicking 'Got it' at the end."""
    if "demo_phase" not in st.session_state:
        pitch, pitch_types = get_demo_content()
        st.session_state.demo_phase      = "animation"
        st.session_state.demo_pitch      = pitch
        st.session_state.demo_types      = pitch_types
        st.session_state.demo_phase_start = time.time()

    pitch       = st.session_state.demo_pitch
    pitch_types = st.session_state.demo_types
    phase       = st.session_state.demo_phase
    elapsed     = time.time() - st.session_state.demo_phase_start

    def _advance(next_phase):
        st.session_state.demo_phase       = next_phase
        st.session_state.demo_phase_start = time.time()
        st.rerun()

    st.title("How to Play")

    if phase == "animation":
        st_autorefresh(interval=100, key="demo_anim_refresh")
        st.markdown("**Step 1 — Watch the pitch fly in.**")
        st.caption("The real 20Hz ball-tracking data animates the pitch from the mound to the plate.")
        components.html(_canvas_html(pitch, "throw"), height=470)
        if elapsed >= ANIM_DURATION_MS / 1000 + 0.8:
            _advance("pitch_type")

    elif phase == "pitch_type":
        st_autorefresh(interval=100, key="demo_pt_refresh")
        remaining = max(0.0, COUNTDOWN_S - elapsed)
        st.markdown("**Step 2 — Identify the pitch type.** You have 3 seconds.")
        if remaining > 0:
            st.progress(remaining / COUNTDOWN_S, text=f"⏱ {remaining:.1f}s")
            n_cols = min(len(pitch_types), 3)
            cols   = st.columns(n_cols)
            for i, pt in enumerate(pitch_types):
                color = PITCH_COLORS.get(pt, "#888")
                with cols[i % n_cols]:
                    st.markdown(
                        f'<div style="height:5px;background:{color};border-radius:3px;margin-bottom:4px"></div>',
                        unsafe_allow_html=True,
                    )
                    st.button(pt, key=f"demo_pt_{pt}", disabled=True, use_container_width=True)
        else:
            correct = pitch["pitch_type_label"]
            st.success(f"That was a **{correct}**!", icon="✅")
            if elapsed >= COUNTDOWN_S + 1.5:
                _advance("swing_take")

    elif phase == "swing_take":
        st_autorefresh(interval=100, key="demo_st_refresh")
        remaining = max(0.0, COUNTDOWN_S - elapsed)
        correct   = pitch["pitch_type_label"]
        st.markdown(f"**Step 3 — Swing or Take?** You called it a {correct}. 3 seconds to decide.")
        if remaining > 0:
            st.progress(remaining / COUNTDOWN_S, text=f"⏱ {remaining:.1f}s")
            c1, c2 = st.columns(2)
            c1.button("🪃 Swing", disabled=True, use_container_width=True)
            c2.button("🤚 Take",  disabled=True, use_container_width=True)
        else:
            in_zone = pitch["zone"] == "in_zone"
            call    = "SWING" if in_zone else "TAKE"
            reason  = "it's in the zone — make contact!" if in_zone else "it's a ball — let it go!"
            st.success(f"Correct call: **{call}** — {reason}", icon="✅")
            if elapsed >= COUNTDOWN_S + 1.5:
                _advance("result")

    elif phase == "result":
        in_zone          = pitch["zone"] == "in_zone"
        correct_decision = True  # demo always shows the right call
        components.html(_canvas_html(pitch, "reveal", correct_decision), height=470)
        st.markdown("**Step 4 — See the result.** The zone box and landing dot appear after your decision.")
        st.info(
            "Each pitch is real tracking data. Correct reads add up — earn a **walk** "
            "with 4 balls, **put the ball in play** with a correct swing on a strike, "
            "or **strike out** after 3 misses."
        )
        if st.button("Got it — face your pitcher →", type="primary"):
            st.session_state.phase = "pitcher_select"
            st.rerun()


def render_start():
    st.title("Your Pitcher")

    eligible = load_eligible_pitchers()
    pitcher_id = random.choice(eligible)

    arsenal = load_arsenals()
    pitcher_arsenal = arsenal[arsenal["pitcher"] == pitcher_id].sort_values(
        "usage_pct", ascending=False
    )
    throws = pitcher_arsenal["throws"].iloc[0]
    total  = pitcher_arsenal["total_pitches"].iloc[0]
    n_types = len(pitcher_arsenal)
    hand_label = "RHP" if throws == "R" else "LHP"

    st.markdown(
        f"**{hand_label}** · {total} pitches tracked · {n_types} pitch types"
    )

    pool = load_pitcher_pools()[pitcher_id]
    components.html(_arsenal_canvas_html(pool), height=470)

    # Compact stats row beneath the canvas
    cols = st.columns(n_types)
    for col, (_, r) in zip(cols, pitcher_arsenal.iterrows()):
        color = PITCH_COLORS.get(r["pitch_type_label"], "#888")
        col.markdown(
            f'<div style="border-left:4px solid {color};padding-left:6px;">'
            f'<b>{r["pitch_type_label"]}</b><br>'
            f'{r["usage_pct"]*100:.0f}% · {r["avg_release_speed_mph"]:.0f} mph'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("⚾ Face This Pitcher", type="primary", use_container_width=True):
        init_game(pitcher_id)
    if c2.button("🔀 Pick Different Pitcher", use_container_width=True):
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

    n_cols = min(len(pitch_types), 3)
    cols = st.columns(n_cols)
    for i, pt in enumerate(pitch_types):
        color = PITCH_COLORS.get(pt, "#888")
        with cols[i % n_cols]:
            st.markdown(
                f'<div style="height:5px;background:{color};'
                f'border-radius:3px;margin-bottom:4px"></div>',
                unsafe_allow_html=True,
            )
            if st.button(pt, key=f"pt_{pt}", use_container_width=True):
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

    def _badge(label):
        c = PITCH_COLORS.get(label, "#888")
        return (f'<span style="background:{c};color:#fff;padding:2px 7px;'
                f'border-radius:4px;font-size:12px;font-weight:600">{label}</span>')

    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr>' +
        "".join(f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #444">{h}</th>'
                for h in ["#", "Pitch", "Guessed", "ID", "Velo", "Zone", "Decision", "S/T", "Outcome"]) +
        "</tr></thead><tbody>"
    )
    for i, h in enumerate(history):
        id_mark = "✅" if h["correct_id"] else "❌"
        st_mark = "✅" if h["outcome"] in ("ball", "in_play") else "❌"
        cells = [
            str(i + 1),
            _badge(h["pitch_type_label"]),
            _badge(h["id_guess"]) if h["id_guess"] else "<i>timeout</i>",
            id_mark,
            f"{h['release_speed_mph']:.0f} mph",
            "Strike" if h["zone"] == "in_zone" else "Ball",
            h["swing_take"] or "<i>timeout</i>",
            st_mark,
            h["outcome"].replace("_", " ").title(),
        ]
        table_html += "<tr>" + "".join(
            f'<td style="padding:4px 8px;border-bottom:1px solid #333">{c}</td>'
            for c in cells
        ) + "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    if st.button("Play Again", type="primary"):
        for key in ["pitcher_id", "pitch_pool", "pitch_idx", "phase",
                    "balls", "strikes", "history", "id_guess", "swing_take",
                    "last_outcome", "last_correct_id", "phase_start"]:
            st.session_state.pop(key, None)
        st.session_state.phase = "pitcher_select"
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Guess the Pitch", page_icon="⚾", layout="centered")

    phase = st.session_state.get("phase")

    if phase is None:
        render_demo()
    elif phase == "pitcher_select":
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
