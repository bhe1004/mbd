"""
Side-by-side animated comparison on the nonholonomic unicycle parking task (3 controllers):
  Ours      = MPPI with BILINEAR Koopman surrogate (captures v*cos/sin theta)
  MPPI-true = MPPI with the TRUE dynamics as rollout model  (oracle upper bound, baseline (a))
  MPPI-DK   = MPPI with LINEAR  Koopman surrogate (z+ = A z + B u)
Same start/goal/cost/samples/smoothing; only the internal rollout MODEL differs.
Each run STOPS when it converges to the goal (pose tolerance).
Smoothing (all runs): control-rate penalty + colored noise + Savitzky-Golay filter.
Outputs: out/unicycle_compare.gif , out/unicycle_compare.png
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
DT = kc.DT

def f(s, u): return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
def step(s, u):
    k1 = f(s, u); k2 = f(s+0.5*DT*k1, u); k3 = f(s+0.5*DT*k2, u); k4 = f(s+DT*k3, u)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def f_batch(S, U):                                   # S:(Ks,3) U:(Ks,2)
    return np.stack([U[:, 0]*np.cos(S[:, 2]), U[:, 0]*np.sin(S[:, 2]), U[:, 1]], axis=1)
def step_batch(S, U):
    k1 = f_batch(S, U); k2 = f_batch(S+0.5*DT*k1, U); k3 = f_batch(S+0.5*DT*k2, U); k4 = f_batch(S+DT*k3, U)
    return S + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi(s): return np.array([s[0], s[1], s[2], np.sin(s[2]), np.cos(s[2]) - 1.0])
def angdiff(a, b): return (a - b + np.pi) % (2*np.pi) - np.pi
N = 5; C = np.zeros((3, N)); C[0, 0] = C[1, 1] = C[2, 2] = 1.0

# ----- identify both Koopman surrogates from the same data -----
def gather(n, L, vb, wb, pb, seed):
    r = np.random.default_rng(seed); X, U, Y = [], [], []
    for _ in range(n):
        s0 = np.array([r.uniform(-pb, pb), r.uniform(-pb, pb), r.uniform(-np.pi, np.pi)])
        us = np.stack([r.uniform(-vb, vb, L), r.uniform(-wb, wb, L)], axis=1)
        ss = [s0.copy()]
        for u in us: ss.append(step(ss[-1], u))
        for k in range(L):
            X.append(Psi(ss[k])); U.append(us[k]); Y.append(Psi(ss[k+1]))
    return np.array(X).T, np.array(U).T, np.array(Y).T
Zx, Uu, Zy = gather(300, 40, 1.5, 1.5, 2.0, seed=1)
M = kc.identify(Zx, Uu, Zy)
A, B0, B1, B2 = M["A"], M["B0"], M["Bs"][0], M["Bs"][1]
Al, Bl = M["A_lin"], M["B_lin"]

STEPS = 170
def mppi_run(mode, s0, goal, T=50, Ks=2048, lam=0.5, sig=np.array([0.5, 0.7]),
             w_du=1.2, tol_pos=0.06, tol_th=0.15, seed=3):
    r = np.random.default_rng(seed); s = s0.copy(); U = np.zeros((T, 2))
    traj = [s.copy()]; ctrls = []; u_prev = np.zeros(2); g = goal; stop = STEPS
    for t in range(STEPS):
        white = r.normal(0, 1, size=(Ks, T, 2))
        eps = uniform_filter1d(white, size=5, axis=1, mode="nearest")*np.sqrt(5.0)*sig   # colored noise
        V = U[None] + eps; cost = np.zeros(Ks)
        if mode == "true": St = np.tile(s, (Ks, 1))
        else:              Zb = np.tile(Psi(s), (Ks, 1))
        for k in range(T):
            uk = V[:, k, :]
            if mode == "bilinear":
                Zb = Zb@A.T + uk@B0.T + uk[:, 0:1]*(Zb@B1.T) + uk[:, 1:2]*(Zb@B2.T); sk = Zb@C.T
            elif mode == "linear":
                Zb = Zb@Al.T + uk@Bl.T; sk = Zb@C.T
            else:                                                       # true dynamics (oracle)
                St = step_batch(St, uk); sk = St
            cost += 3.0*((sk[:, 0]-g[0])**2 + (sk[:, 1]-g[1])**2) + 0.15*(sk[:, 2]-g[2])**2 + 0.03*(uk**2).sum(1)
            du = uk - (u_prev if k == 0 else V[:, k-1, :]); cost += w_du*(du**2).sum(1)   # control-rate penalty
        cost += 40.0*((sk[:, 0]-g[0])**2 + (sk[:, 1]-g[1])**2) + 8.0*(sk[:, 2]-g[2])**2
        w = np.exp(-(cost-cost.min())/lam); w /= w.sum()
        U = U + np.einsum("k,ktd->td", w, eps)
        U = savgol_filter(U, window_length=11, polyorder=3, axis=0)      # SG smoothing
        u0 = np.clip(U[0], [-1.5, -1.5], [1.5, 1.5]); s = step(s, u0); u_prev = u0
        traj.append(s.copy()); ctrls.append(u0.copy())
        U = np.roll(U, -1, axis=0); U[-1] = U[-2]
        if np.linalg.norm(s[:2]-g[:2]) < tol_pos and abs(angdiff(s[2], g[2])) < tol_th:
            stop = len(traj)-1; break                                   # converged -> STOP
    return np.array(traj), np.array(ctrls), stop

s0 = np.array([1.5, 1.5, -np.pi/2]); goal = np.array([0.0, 0.0, 0.0])
runs = [("Ours: bilinear Koopman MPPI", "bilinear", "C0"),
        ("MPPI-true: oracle (true dynamics)", "true", "C2"),
        ("MPPI-DK: linear Koopman MPPI", "linear", "C3")]
jerk = lambda c: float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1))) if len(c) > 1 else 0.0
results = []
for title, mode, col in runs:
    tr, c, stop = mppi_run(mode, s0, goal)
    results.append((title, tr, col, stop))
    conv = f"converged@step {stop}" if stop < STEPS else "did NOT converge"
    print(f"[{mode:8s}] final ||pos||={np.linalg.norm(tr[-1,:2]):.3f}  {conv}  mean|du|={jerk(c):.4f}")

# ----- 3-panel animation -----
lim = 2.6
def robot(ax, s, c):
    ax.arrow(s[0], s[1], 0.35*np.cos(s[2]), 0.35*np.sin(s[2]), head_width=0.16, head_length=0.16,
             fc=c, ec=c, lw=2, length_includes_head=True, zorder=5)
def setup(ax, title):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10.5)
    ax.scatter([goal[0]], [goal[1]], c="k", marker="*", s=240, zorder=4, label="goal")
    ax.arrow(goal[0], goal[1], 0.3, 0.0, head_width=0.12, head_length=0.12, fc="k", ec="k", zorder=4)
    ax.scatter([s0[0]], [s0[1]], c="0.5", marker="o", s=70, zorder=4, label="start")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.7))
nframes = max(len(r[1]) for r in results)
def update(i):
    for ax, (title, tr, col, stop) in zip(axes, results):
        ax.clear(); setup(ax, title)
        j = min(i, len(tr)-1)
        ax.plot(tr[:j+1, 0], tr[:j+1, 1], "-", c=col, lw=2, alpha=0.9)
        robot(ax, tr[j], col)
        tag = "  PARKED" if (stop < STEPS and j >= stop) else ""
        ax.text(-lim+0.1, lim-0.25, f"step {j:3d}  dist {np.linalg.norm(tr[j,:2]-goal[:2]):.2f}{tag}",
                fontsize=9, family="monospace")
        ax.legend(loc="lower right", fontsize=8)
    return []
ani = FuncAnimation(fig, update, frames=nframes, interval=70, blit=False)
ani.save(os.path.join(OUT, "unicycle_compare.gif"), writer=PillowWriter(fps=15))
update(nframes-1); plt.tight_layout(); plt.savefig(os.path.join(OUT, "unicycle_compare.png"), dpi=110)
print(f"outputs -> {OUT}/unicycle_compare.gif , unicycle_compare.png")
