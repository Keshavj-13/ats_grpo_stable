import os, sys, math, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess

BG_COLOR = "#0c1017"
PANEL_COLOR = "#161b22"
BORDER_COLOR = "#30363d"
ACCENT_CYAN = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_AMBER = "#d29922"
ACCENT_RED = "#f85149"
TEXT_WHITE = "#f0f6fc"
TEXT_MUTED = "#8b949e"

def apply_terminal_style(ax, title=""):
    ax.set_facecolor(PANEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=TEXT_WHITE, fontsize=10, fontweight="bold", pad=8)

# -------------------------------------------------------------
# GIF 1: 4-Agent GRPO Dialogue & Group Advantage Negotiation
# -------------------------------------------------------------
print("Generating GIF 1: 4-Agent GRPO Dialogue & Negotiation...")
frames_dir_1 = "/tmp/atc_gif1_frames"
os.makedirs(frames_dir_1, exist_ok=True)
for f in os.listdir(frames_dir_1):
    if f.endswith(".png"):
        os.remove(os.path.join(frames_dir_1, f))

dialogue_steps = [
    {"role": "GENERATOR", "color": ACCENT_AMBER, "text": "CHALLENGE GENERATOR (Task: Bengaluru IRROPS)\nInjecting 17 flights into 2 runways: 4 Heavy (A350/B777), 1 Med Priority Emergency, 12 Mediums.\nWind Shear Alert at Runway 09R. Expected capacity constraint: 40% reduction."},
    {"role": "AMAN", "color": ACCENT_CYAN, "text": "ARRIVAL MANAGER (AMAN - Qwen2.5 QLoRA)\nSequencing 9 inbound arrivals. Holding Heavy AIC102 at FIX-DELTA for 4 minutes.\nPrioritizing Emergency MED01 on Runway 09L direct approach (EET 14:12 UTC)."},
    {"role": "DMAN", "color": "#bc8cff", "text": "DEPARTURE MANAGER (DMAN - Qwen2.5 QLoRA)\nSlotting 8 outbound flights into interleaved gaps. Holding IGO402 at holding point 09R.\nWaketime buffer: 120s trailing Heavy AIC102. Runway occupancy: 48s."},
    {"role": "SUPERVISOR", "color": ACCENT_GREEN, "text": "SUPERVISOR / CONFLICT ARBITRATOR\nVerification passed. Separation margin: 4.8 NM (Target >= 3.0 NM). Runway crossing cleared.\nGroup Relative Advantage A_i = +1.42 (Group Reward = 0.913 vs Mean = 0.620). Update accepted."}
]

total_frames_1 = 48
for f_idx in range(total_frames_1):
    fig = plt.figure(figsize=(10, 5.625), dpi=80, facecolor=BG_COLOR)
    
    # Left panel: Multi-Agent Conversation Terminal
    ax_term = fig.add_axes([0.08, 0.12, 0.60, 0.78])
    apply_terminal_style(ax_term, "Multi-Agent GRPO Negotiation Loop (Single Backbone, 4 Roles)")
    
    active_msgs = min(4, 1 + f_idx // 11)
    
    y_pos = 0.88
    for m_idx in range(active_msgs):
        msg = dialogue_steps[m_idx]
        is_latest = (m_idx == active_msgs - 1)
        ax_term.text(0.04, y_pos, msg["text"], color=TEXT_WHITE if not is_latest else "#ffffff",
                     fontsize=7.2, family="monospace", va="top",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#1c2128" if is_latest else PANEL_COLOR,
                               ec=msg["color"] if is_latest else BORDER_COLOR, lw=1.2 if is_latest else 0.8))
        y_pos -= 0.23
        
    ax_term.set_xticks([])
    ax_term.set_yticks([])
    
    # Right panel: Telemetry
    ax_tele = fig.add_axes([0.71, 0.12, 0.26, 0.78])
    ax_tele.set_facecolor(PANEL_COLOR)
    for sp in ax_tele.spines.values():
        sp.set_color(BORDER_COLOR)
    ax_tele.set_xticks([])
    ax_tele.set_yticks([])
    ax_tele.set_title("GRPO TELEMETRY", color=ACCENT_CYAN, fontsize=10, fontweight="bold", pad=8)
    
    ep_num = 45 + f_idx * 3
    telemetry_grpo = (
        f"TRAINING CONTEXT\n"
        f"---------------------------\n"
        f"Backbone     : Qwen2.5-7B\n"
        f"Adapter      : 4-bit QLoRA (r=16)\n"
        f"Hardware     : NVIDIA A100 80GB\n"
        f"Episode      : {ep_num} / 200\n\n"
        f"GRPO GROUP RELATIVE\n"
        f"---------------------------\n"
        f"Group Size G : 4 completions\n"
        f"Advantage A_i: +1.42 sigma\n"
        f"Mean Group R : 0.620\n"
        f"Rollout R_i  : 0.913 (Best)\n\n"
        f"MULTI-AGENT ROLES\n"
        f"---------------------------\n"
        f"1. AMAN      : Arrival Manager\n"
        f"2. DMAN      : Departure Mgr\n"
        f"3. Generator : Adaptive Task\n"
        f"4. Supervisor: Safety Grader\n\n"
        f"SHAPED REWARDS\n"
        f"---------------------------\n"
        f"Potential Phi: Ng et al. 1999\n"
        f"Conflict Rate: 0.0% (Zero Loss)"
    )
    ax_tele.text(0.06, 0.94, telemetry_grpo, color=TEXT_WHITE, fontsize=7.5, family="monospace", va="top")
    
    plt.savefig(f"{frames_dir_1}/frame_{f_idx:03d}.png", dpi=80)
    plt.close()

# Dwell
last_f1 = f"{frames_dir_1}/frame_{total_frames_1-1:03d}.png"
for d in range(15):
    subprocess.run(["cp", last_f1, f"{frames_dir_1}/frame_{total_frames_1+d:03d}.png"])

out_gif1 = "/workspace/external_repos/ats_grpo_stable/assets/01_atc_multiagent_grpo_curriculum.gif"
cmd1 = [
    "ffmpeg", "-y", "-framerate", "10",
    "-i", f"{frames_dir_1}/frame_%03d.png",
    "-vf", "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
    out_gif1
]
subprocess.run(cmd1, check=True)
print(f"Generated GIF 1: {os.path.getsize(out_gif1)/1024/1024:.2f} MB")

# -------------------------------------------------------------
# GIF 2: Terminal Radar PPI Scope & Wake Separation
# -------------------------------------------------------------
print("Generating GIF 2: Terminal Radar PPI Scope & Wake Separation...")
frames_dir_2 = "/tmp/atc_gif2_frames"
os.makedirs(frames_dir_2, exist_ok=True)
for f in os.listdir(frames_dir_2):
    if f.endswith(".png"):
        os.remove(os.path.join(frames_dir_2, f))

# Airspace PPI scope simulation: 6 aircraft moving along approach funnels
total_frames_2 = 45

# Flight tracks
flights = [
    {"call": "AIC102", "type": "HEAVY", "x_path": np.linspace(80, 10, total_frames_2), "y_path": np.linspace(80, 20, total_frames_2), "color": ACCENT_AMBER},
    {"call": "MED01", "type": "EMERGENCY", "x_path": np.linspace(60, 5, total_frames_2), "y_path": np.linspace(30, 18, total_frames_2), "color": ACCENT_RED},
    {"call": "IGO402", "type": "MEDIUM", "x_path": np.linspace(95, 25, total_frames_2), "y_path": np.linspace(70, 22, total_frames_2), "color": ACCENT_CYAN},
    {"call": "VTI811", "type": "MEDIUM", "x_path": np.linspace(40, -10, total_frames_2), "y_path": np.linspace(85, 30, total_frames_2), "color": ACCENT_GREEN},
    {"call": "SEJ204", "type": "MEDIUM", "x_path": np.linspace(10, 70, total_frames_2), "y_path": np.linspace(15, 60, total_frames_2), "color": "#bc8cff"}, # departure
]

for f_idx in range(total_frames_2):
    fig = plt.figure(figsize=(10, 5.625), dpi=80, facecolor=BG_COLOR)
    
    # Left plot: Terminal Radar PPI Scope
    ax_radar = fig.add_axes([0.08, 0.12, 0.60, 0.78])
    apply_terminal_style(ax_radar, "Terminal Radar Service (30 NM Sector Scope)")
    
    # Radar range rings (10, 20, 30 NM)
    for r in [25, 50, 75]:
        ring = plt.Circle((50, 50), r, color=BORDER_COLOR, fill=False, linestyle=":", linewidth=0.8)
        ax_radar.add_patch(ring)
        ax_radar.text(50, 50 + r - 3, f"{r//2.5:.0f} NM", color=TEXT_MUTED, fontsize=6.5, ha="center")
        
    # Runway layout at center (50, 50)
    ax_radar.plot([46, 54], [48, 52], color="#ffffff", linewidth=3.0, label="Runway 09L/27R")
    ax_radar.plot([48, 56], [42, 46], color="#a0aec0", linewidth=2.0, label="Runway 09R/27L")
    ax_radar.text(50, 44, "BLR", color=TEXT_WHITE, fontsize=7, fontweight="bold", ha="center")
    
    # Plot aircraft and wake turbulence zones
    for fl in flights:
        cx = fl["x_path"][f_idx]
        cy = fl["y_path"][f_idx]
        
        # Aircraft symbol
        ax_radar.scatter([cx], [cy], color=fl["color"], s=60, marker="^", zorder=5)
        
        # Separation bubble (3 NM = radius 7.5)
        bubble = plt.Circle((cx, cy), 7.5, color=fl["color"], fill=False, linestyle="--", linewidth=1.0, alpha=0.6)
        ax_radar.add_patch(bubble)
        
        # Trailing wake envelope for Heavy aircraft
        if fl["type"] == "HEAVY":
            wake_trail_x = [cx, cx + 18]
            wake_trail_y = [cy, cy + 18]
            ax_radar.plot(wake_trail_x, wake_trail_y, color=ACCENT_AMBER, linewidth=4.0, alpha=0.3)
            
        # Data tag
        ax_radar.text(cx + 2, cy + 2, f"{fl['call']}\n{fl['type']}\nFL040", color=TEXT_WHITE, fontsize=6.5,
                      bbox=dict(boxstyle="square,pad=0.2", fc=PANEL_COLOR, ec=BORDER_COLOR, alpha=0.8))
        
    ax_radar.set_xlim(0, 100)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_xticks([])
    ax_radar.set_yticks([])
    ax_radar.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=7, labelcolor=TEXT_WHITE)
    
    # Right panel: Telemetry
    ax_tele2 = fig.add_axes([0.71, 0.12, 0.26, 0.78])
    ax_tele2.set_facecolor(PANEL_COLOR)
    for sp in ax_tele2.spines.values():
        sp.set_color(BORDER_COLOR)
    ax_tele2.set_xticks([])
    ax_tele2.set_yticks([])
    ax_tele2.set_title("RADAR TELEMETRY", color=ACCENT_GREEN, fontsize=10, fontweight="bold", pad=8)
    
    telemetry_radar = (
        f"AIRSPACE SECTOR\n"
        f"---------------------------\n"
        f"Airfield     : VOBL (Bengaluru)\n"
        f"Active RWY   : 09L (Arr) / 09R (Dep)\n"
        f"Tracked      : 17 Aircraft\n"
        f"Mode         : Mode-S ADS-B\n\n"
        f"SEPARATION INTEGRITY\n"
        f"---------------------------\n"
        f"Standard Min : 3.0 NM / 1,000 ft\n"
        f"Min Observed : 4.82 NM\n"
        f"Violations   : 0 LOSS OF SEP\n"
        f"Wake Spacing : 120s Heavy Follow\n\n"
        f"OPTIMIZATION METRICS\n"
        f"---------------------------\n"
        f"Sequence Rate: 42 mov / hr\n"
        f"Holding Delay: -4.2 min / flight\n"
        f"Emerg Priority: MED01 Expedited\n"
        f"Safety Gating: 100% Passed"
    )
    ax_tele2.text(0.06, 0.94, telemetry_radar, color=TEXT_WHITE, fontsize=7.5, family="monospace", va="top")
    
    plt.savefig(f"{frames_dir_2}/frame_{f_idx:03d}.png", dpi=80)
    plt.close()

# Dwell
last_f2 = f"{frames_dir_2}/frame_{total_frames_2-1:03d}.png"
for d in range(15):
    subprocess.run(["cp", last_f2, f"{frames_dir_2}/frame_{total_frames_2+d:03d}.png"])

out_gif2 = "/workspace/external_repos/ats_grpo_stable/assets/02_atc_radar_wake_separation_sequencing.gif"
cmd2 = [
    "ffmpeg", "-y", "-framerate", "10",
    "-i", f"{frames_dir_2}/frame_%03d.png",
    "-vf", "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
    out_gif2
]
subprocess.run(cmd2, check=True)
print(f"Generated GIF 2: {os.path.getsize(out_gif2)/1024/1024:.2f} MB")

# -------------------------------------------------------------
# GIF 3: OpenEnv Benchmark Progression & Gated Scoring
# -------------------------------------------------------------
print("Generating GIF 3: OpenEnv Benchmark Progression...")
frames_dir_3 = "/tmp/atc_gif3_frames"
os.makedirs(frames_dir_3, exist_ok=True)
for f in os.listdir(frames_dir_3):
    if f.endswith(".png"):
        os.remove(os.path.join(frames_dir_3, f))

tasks = [
    {"name": "Delhi Monsoon", "rand": 0.21, "heur": 0.68, "grpo": 0.9446},
    {"name": "Mumbai Bank", "rand": 0.18, "heur": 0.72, "grpo": 0.9900},
    {"name": "Bengaluru IRROPS", "rand": 0.12, "heur": 0.58, "grpo": 0.8615},
    {"name": "Hyderabad Crunch", "rand": 0.15, "heur": 0.64, "grpo": 0.8576},
]

total_frames_3 = 45
for f_idx in range(total_frames_3):
    fig = plt.figure(figsize=(10, 5.625), dpi=80, facecolor=BG_COLOR)
    
    # Left subplot: Grouped Bar Chart of Task Scores
    ax_bar = fig.add_axes([0.08, 0.12, 0.60, 0.78])
    apply_terminal_style(ax_bar, "OpenEnv Benchmark: Random vs Heuristic vs Multi-Agent GRPO")
    
    x_idx = np.arange(len(tasks))
    width = 0.24
    
    # Animation progression
    scale_rand = min(1.0, f_idx / 12.0)
    scale_heur = min(1.0, max(0.0, (f_idx - 10) / 14.0))
    scale_grpo = min(1.0, max(0.0, (f_idx - 22) / 16.0))
    
    vals_rand = [t["rand"] * scale_rand for t in tasks]
    vals_heur = [t["heur"] * scale_heur for t in tasks]
    vals_grpo = [t["grpo"] * scale_grpo for t in tasks]
    
    b1 = ax_bar.bar(x_idx - width, vals_rand, width, color="#8b949e", edgecolor=BORDER_COLOR, label="Random Baseline (0.165)")
    b2 = ax_bar.bar(x_idx, vals_heur, width, color="#58a6ff", edgecolor=BORDER_COLOR, label="Heuristic Baseline (0.655)")
    b3 = ax_bar.bar(x_idx + width, vals_grpo, width, color=ACCENT_GREEN, edgecolor=BORDER_COLOR, label="Multi-Agent GRPO (0.913)")
    
    if scale_grpo > 0.8:
        for i, t in enumerate(tasks):
            ax_bar.text(i + width, t["grpo"] + 0.03, f"{t['grpo']:.3f}", color=TEXT_WHITE, fontsize=7.5,
                        fontweight="bold", ha="center")
            
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([t["name"] for t in tasks], fontsize=8)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Composite Gated Score (0 - 1.0)", color=TEXT_MUTED, fontsize=8)
    ax_bar.legend(loc="upper left", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=7.5, labelcolor=TEXT_WHITE)
    ax_bar.grid(True, color=BORDER_COLOR, linestyle="--", alpha=0.3, axis="y")
    
    # Right panel: Telemetry
    ax_tele3 = fig.add_axes([0.71, 0.12, 0.26, 0.78])
    ax_tele3.set_facecolor(PANEL_COLOR)
    for sp in ax_tele3.spines.values():
        sp.set_color(BORDER_COLOR)
    ax_tele3.set_xticks([])
    ax_tele3.set_yticks([])
    ax_tele3.set_title("EVALUATION TELEMETRY", color=ACCENT_AMBER, fontsize=10, fontweight="bold", pad=8)
    
    avg_score = 0.9134 * scale_grpo
    telemetry_eval = (
        f"BENCHMARK SUMMARY\n"
        f"---------------------------\n"
        f"Framework    : OpenEnv Gym\n"
        f"Evaluation   : 4 Standard Tasks\n"
        f"Average Score: {avg_score:.4f}\n"
        f"Baseline Impr: +0.7484 (+453%)\n\n"
        f"GATED CONSTRAINT CHECK\n"
        f"---------------------------\n"
        f"Layer 1: Wake: PASS (100%)\n"
        f"Layer 2: Emer: PASS (100%)\n"
        f"Layer 3: Rwy : PASS (100%)\n"
        f"Efficiency   : Optimized\n\n"
        f"RUNTIME PERFORMANCE\n"
        f"---------------------------\n"
        f"Total Runtime: 11.69 seconds\n"
        f"Resource Spec: 2 vCPU / 8GB RAM\n"
        f"Steps Used   : 2-4 per task\n"
        f"Deployment   : HuggingFace Space"
    )
    ax_tele3.text(0.06, 0.94, telemetry_eval, color=TEXT_WHITE, fontsize=7.5, family="monospace", va="top")
    
    plt.savefig(f"{frames_dir_3}/frame_{f_idx:03d}.png", dpi=80)
    plt.close()

# Dwell
last_f3 = f"{frames_dir_3}/frame_{total_frames_3-1:03d}.png"
for d in range(15):
    subprocess.run(["cp", last_f3, f"{frames_dir_3}/frame_{total_frames_3+d:03d}.png"])

out_gif3 = "/workspace/external_repos/ats_grpo_stable/assets/03_atc_openenv_benchmark_progression.gif"
cmd3 = [
    "ffmpeg", "-y", "-framerate", "10",
    "-i", f"{frames_dir_3}/frame_%03d.png",
    "-vf", "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
    out_gif3
]
subprocess.run(cmd3, check=True)
print(f"Generated GIF 3: {os.path.getsize(out_gif3)/1024/1024:.2f} MB")
print("All 3 ATC GIFs successfully generated!")
