"""
Pendulum BALANCING (inverted stabilization / disturbance rejection), 3 controllers:
  Ours      = MPPI with BILINEAR Koopman surrogate
  MPPI-true = MPPI with TRUE dynamics rollout (oracle baseline (a))
  MPPI-DK   = MPPI with LINEAR  Koopman surrogate
theta=0 is UPRIGHT (unstable). Pendulum is kicked from upright (theta=0, dtheta0>0); the
controller must CATCH it and re-stabilize to upright. Same cost/samples/smoothing; only the
internal rollout MODEL differs. Each run stops when balanced (upright & slow).
Lifting Psi=[sin th, cos th-1, dth, sin th*dth, cos th*dth, dth^2]; cross terms -> bilinear coupling.
NOTE: full SWING-UP (from hanging down) needs a long MPPI horizon over which the simple EDMDc
Koopman rollout error compounds badly -> only the oracle pumps; the lifted models stay passive.
That long-horizon Koopman-rollout limitation is itself a finding (needs deep-Koopman / energy-shaping).
Outputs: out/pendulum_compare.gif , out/pendulum_compare.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import koopman_core as kc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

GL, B, ML2, DT, UMAX = 10.0, 0.1, 1.0, 0.02, 8.0     # g/l, damping, m l^2, dt, torque limit
def f(s, u): return np.array([s[1], GL*np.sin(s[0]) - B*s[1] + u/ML2])   # theta=0 upright (unstable)
def step(s, u):
    k1 = f(s, u); k2 = f(s+0.5*DT*k1, u); k3 = f(s+0.5*DT*k2, u); k4 = f(s+DT*k3, u)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def f_batch(S, U):
    return np.stack([S[:, 1], GL*np.sin(S[:, 0]) - B*S[:, 1] + U/ML2], axis=1)
def step_batch(S, U):
    k1 = f_batch(S, U); k2 = f_batch(S+0.5*DT*k1, U); k3 = f_batch(S+0.5*DT*k2, U); k4 = f_batch(S+DT*k3, U)
    return S + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi(s):
    th, dth = s[0], s[1]
    return np.array([np.sin(th), np.cos(th)-1.0, dth, np.sin(th)*dth, np.cos(th)*dth, dth*dth])
N = 6; C = np.zeros((3, N)); C[0, 0] = C[1, 1] = C[2, 2] = 1.0    # -> (sin th, cos th-1, dth)

def gather(D, useed):
    # uniform single-transition sampling over the near-upright operating region (clean coverage)
    r = np.random.default_rng(useed)
    TH = r.uniform(-1.4, 1.4, D); DTH = r.uniform(-5, 5, D); US = r.uniform(-UMAX, UMAX, D)
    X = np.array([Psi(np.array([TH[i], DTH[i]])) for i in range(D)]).T
    Y = np.array([Psi(step(np.array([TH[i], DTH[i]]), US[i])) for i in range(D)]).T
    return X, US[None, :], Y
Zx, Uu, Zy = gather(20000, 1)
M = kc.identify(Zx, Uu, Zy)
A, B0, B1 = M["A"], M["B0"], M["Bs"][0]; Al, Bl = M["A_lin"], M["B_lin"]
rl = float(np.sqrt(np.mean(np.sum((Al@Zx + Bl@Uu - Zy)**2, axis=0))))
rb = float(np.sqrt(np.mean(np.sum((kc.predict_bi(M, Zx, Uu) - Zy)**2, axis=0))))
print(f"[id] D={Zx.shape[1]}  1-step lifted RMSE  linear={rl:.4e}  bilinear={rb:.4e}  (lin/bi={rl/rb:.2f}x)")

STEPS = 200
def feats(mode, Zb, St):
    if mode == "true":
        return np.sin(St[:, 0]), np.cos(St[:, 0]) - 1.0, St[:, 1]
    sk = Zb @ C.T
    return sk[:, 0], sk[:, 1], sk[:, 2]

def mppi_run(mode, s0, T=50, Ks=2048, lam=0.8, sig=2.0, w_du=0.3, seed=3):
    r = np.random.default_rng(seed); s = s0.copy(); U = np.zeros(T)
    traj = [s.copy()]; ctrls = []; u_prev = 0.0; stop = STEPS
    for t in range(STEPS):
        white = r.normal(0, 1, size=(Ks, T))
        eps = uniform_filter1d(white, size=5, axis=1, mode="nearest")*np.sqrt(5.0)*sig
        V = U[None] + eps; cost = np.zeros(Ks)
        Zb = np.tile(Psi(s), (Ks, 1)) if mode != "true" else None
        St = np.tile(s, (Ks, 1)) if mode == "true" else None
        for k in range(T):
            uk = V[:, k]
            if mode == "bilinear":
                Zb = Zb@A.T + np.outer(uk, B0[:, 0]) + uk[:, None]*(Zb@B1.T)
            elif mode == "linear":
                Zb = Zb@Al.T + np.outer(uk, Bl[:, 0])
            else:
                St = step_batch(St, uk)
            s0f, s1f, s2f = feats(mode, Zb, St)
            cost += 3.0*(s0f**2 + s1f**2) + 0.05*s2f**2 + 0.005*uk**2           # 2(1-cos th)+...
            if w_du > 0:
                du = uk - (u_prev if k == 0 else V[:, k-1]); cost += w_du*du**2
        s0f, s1f, s2f = feats(mode, Zb, St)
        cost += 60.0*(s0f**2 + s1f**2) + 3.0*s2f**2                            # strong terminal (balance)
        w = np.exp(-(cost-cost.min())/lam); w /= w.sum()
        U = U + np.einsum("k,kt->t", w, eps)
        U = savgol_filter(U, window_length=11, polyorder=3)
        u0 = float(np.clip(U[0], -UMAX, UMAX)); s = step(s, u0); u_prev = u0
        traj.append(s.copy()); ctrls.append(u0)
        U = np.roll(U, -1); U[-1] = U[-2]
        if (1.0 - np.cos(s[0])) < 0.015 and abs(s[1]) < 0.3:                   # upright & slow -> STOP
            stop = len(traj)-1; break
    return np.array(traj), np.array(ctrls), stop

s0 = np.array([0.0, 1.2])                         # small kick from upright (disturbance rejection)
runs = [("Ours: bilinear Koopman MPPI", "bilinear", "C0"),
        ("MPPI-true: oracle (true dynamics)", "true", "C2"),
        ("MPPI-DK: linear Koopman MPPI", "linear", "C3")]
jerk = lambda c: float(np.mean(np.abs(np.diff(c)))) if len(c) > 1 else 0.0
results = []
for title, mode, col in runs:
    tr, c, stop = mppi_run(mode, s0)
    up = float(1.0 - np.cos(tr[-1, 0]))
    conv = f"balanced@step {stop}" if stop < STEPS else "did NOT balance"
    results.append((title, tr, col, stop))
    print(f"[{mode:8s}] final 1-cos(th)={up:.3f} |dth|={abs(tr[-1,1]):.2f}  {conv}  mean|du|={jerk(c):.4f}")

# ----- 3-panel pendulum animation -----
def setup(ax, title):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10.5)
    ax.scatter([0], [1], c="k", marker="*", s=160, zorder=3)            # upright target
    ax.scatter([0], [0], c="0.4", s=40, zorder=4)                       # pivot
def draw(ax, tr, j, col):
    th = tr[j, 0]; bx, by = np.sin(th), np.cos(th)
    tail = max(0, j-40)
    ax.plot(np.sin(tr[tail:j+1, 0]), np.cos(tr[tail:j+1, 0]), "-", c=col, lw=1, alpha=0.35)
    ax.plot([0, bx], [0, by], "-", c=col, lw=3, zorder=5)               # rod
    ax.scatter([bx], [by], c=col, s=130, zorder=6)                      # bob

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
nframes = max(len(r[1]) for r in results)
def update(i):
    for ax, (title, tr, col, stop) in zip(axes, results):
        ax.clear(); setup(ax, title)
        j = min(i, len(tr)-1); draw(ax, tr, j, col)
        tag = "  BALANCED" if (stop < STEPS and j >= stop) else ""
        ax.text(-1.32, 1.2, f"t={j*DT:4.1f}s  1-cos={1-np.cos(tr[j,0]):.2f}{tag}", fontsize=9, family="monospace")
    return []
ani = FuncAnimation(fig, update, frames=nframes, interval=40, blit=False)
ani.save(os.path.join(OUT, "pendulum_compare.gif"), writer=PillowWriter(fps=25))
update(nframes-1); plt.tight_layout(); plt.savefig(os.path.join(OUT, "pendulum_compare.png"), dpi=110)
print(f"outputs -> {OUT}/pendulum_compare.gif , pendulum_compare.png")
