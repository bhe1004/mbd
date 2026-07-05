"""Generate the two NEW paper figures (no retraining, deterministic):
  (1) fig_arm_bar.png  -- 7-DOF arm per-target final error (ours/oracle/MPPI-DK), from arm_robust.py output.
  (2) fig_tube.png     -- unicycle certified error tube vs true lifted error (same algebra as tier1_unicycle).
Writes straight into the mounted template/figs (/figs)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koopman_core as kc

FIGS = "/figs"
plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.35,
    "axes.labelsize": 13, "legend.fontsize": 11, "figure.dpi": 150,
})

# ========================================================== (1) 7-DOF arm per-target bars
# Authoritative numbers captured from arm_robust.py (seed-fixed, Docker run).
labels = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
ours   = [0.024, 0.022, 0.017, 0.283, 0.043, 0.024, 0.025]
oracle = [0.017, 0.023, 0.018, 0.022, 0.021, 0.023, 0.023]
dk     = [0.681, 0.921, 0.365, 0.630, 0.230, 0.465, 0.979]
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.bar(x - w, ours,   w, label="Ours (bilinear)",    color="#1f77b4")
ax.bar(x,     oracle, w, label="Oracle (true dyn.)", color="#2ca02c")
ax.bar(x + w, dk,     w, label="MPPI-DK (linear)",   color="#d62728")
ax.axhline(0.05, ls="--", color="k", lw=1.2)
ax.annotate("reach threshold (0.05 m)", xy=(6.45, 0.05), xytext=(6.45, 0.14),
            ha="right", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("reach target"); ax.set_ylabel("final EE error [m]")
ax.set_ylim(0, 1.05); ax.legend(loc="upper left", ncol=1)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_arm_bar.png"), dpi=200)
plt.close(fig)
print("[arm bar] saved  ours 6/7  oracle 7/7  MPPI-DK 0/7")

# ========================================================== (2) unicycle certified error tube
DT = kc.DT
def f_ct(s, u): return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
def step(s, u):
    k1 = f_ct(s, u); k2 = f_ct(s+0.5*DT*k1, u); k3 = f_ct(s+0.5*DT*k2, u); k4 = f_ct(s+DT*k3, u)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi(s): return np.array([s[0], s[1], s[2], np.sin(s[2]), np.cos(s[2]) - 1.0])
def gather(n, L, seed):
    r = np.random.default_rng(seed); X, U, Y = [], [], []
    for _ in range(n):
        s0 = np.array([r.uniform(-2, 2), r.uniform(-2, 2), r.uniform(-np.pi, np.pi)])
        us = np.stack([r.uniform(-1.5, 1.5, L), r.uniform(-1.5, 1.5, L)], 1)
        ss = [s0.copy()]
        for u in us: ss.append(step(ss[-1], u))
        ss = np.array(ss)
        for k in range(L):
            X.append(Psi(ss[k])); U.append(us[k]); Y.append(Psi(ss[k+1]))
    return np.array(X).T, np.array(U).T, np.array(Y).T

Zx, Uu, Zy = gather(300, 40, 1)
M = kc.identify(Zx, Uu, Zy)
cx, cu = kc.fit_bound(M, Zx, Uu, Zy)

# representative held-out test trajectory
r = np.random.default_rng(7); L = 40
s0 = np.array([r.uniform(-1.5, 1.5), r.uniform(-1.5, 1.5), r.uniform(-np.pi, np.pi)])
us = np.stack([r.uniform(-1.5, 1.5, L), r.uniform(-1.5, 1.5, L)], 1)
ss = [s0.copy()]
for u in us: ss.append(step(ss[-1], u))
ss = np.array(ss); z0 = Psi(s0)
zt = np.array([Psi(s) for s in ss]); zh = kc.roll_bi(M, z0, us)
dl = np.linalg.norm(zt - zh, axis=1); e = kc.euclid_tube(M, zh, us, cx, cu)

fig, ax = plt.subplots(figsize=(4.7, 3.2))
ax.plot(dl, lw=2.2, color="#1f77b4", label=r"true lifted error $\|z_k-\hat z_k\|$")
ax.plot(e, "--", lw=2.2, color="#d62728", label=r"certified tube $e_k$")
ax.set_yscale("log"); ax.set_xlabel(r"rollout step $k$"); ax.set_ylabel("lifted-space error")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tube.png"), dpi=200)
viol = int(np.sum(dl > e + 1e-9))
print(f"[tube] saved  c_x={cx:.4f} c_u={cu:.4f}  violations={viol}/{len(dl)}")
