"""
Tier-1: nonholonomic unicycle  (state s=(x,y,theta), input u=(v,omega)).
  x' = v cos(theta),  y' = v sin(theta),  theta' = omega
Input enters STRONGLY bilinearly (v multiplies cos/sin theta). Trig-closed lifting
Psi=[x,y,theta,sin theta,cos theta - 1] (Psi(0)=0, C=[I3|0]).

Expect: bilinear EDMDc nearly exact (v*cos theta = u*(feature)) while linear EDMDc
(constant Bu) fails badly -> large C1 advantage. Plus tube check + MPPI parking.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koopman_core as kc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
_log = []
def log(m=""):
    print(m); _log.append(str(m))

DT = kc.DT
def f_ct(s, u):
    return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
def step(s, u):
    k1 = f_ct(s, u); k2 = f_ct(s+0.5*DT*k1, u); k3 = f_ct(s+0.5*DT*k2, u); k4 = f_ct(s+DT*k3, u)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi(s):
    return np.array([s[0], s[1], s[2], np.sin(s[2]), np.cos(s[2]) - 1.0])
N = 5
C = np.zeros((3, N)); C[0, 0] = C[1, 1] = C[2, 2] = 1.0

def gather(n_traj, L, vbox, wbox, pbox, seed):
    r = np.random.default_rng(seed); X, U, Y = [], [], []
    for _ in range(n_traj):
        s0 = np.array([r.uniform(-pbox, pbox), r.uniform(-pbox, pbox), r.uniform(-np.pi, np.pi)])
        us = np.stack([r.uniform(-vbox, vbox, L), r.uniform(-wbox, wbox, L)], axis=1)
        ss = [s0.copy()]
        for u in us: ss.append(step(ss[-1], u))
        ss = np.array(ss)
        for k in range(L):
            X.append(Psi(ss[k])); U.append(us[k]); Y.append(Psi(ss[k+1]))
    return np.array(X).T, np.array(U).T, np.array(Y).T

# --------------------------------------------------------------------- identify
Zx, Uu, Zy = gather(300, 40, 1.5, 1.5, 2.0, seed=1)
M = kc.identify(Zx, Uu, Zy)
cx, cu = kc.fit_bound(M, Zx, Uu, Zy)
rl = float(np.sqrt(np.mean(np.sum((M["A_lin"]@Zx + M["B_lin"]@Uu - Zy)**2, axis=0))))
rb = float(np.sqrt(np.mean(np.sum((kc.predict_bi(M, Zx, Uu) - Zy)**2, axis=0))))
log(f"[id ] N={N} m=2 D={Zx.shape[1]}  1-step lifted RMSE  linear={rl:.4e}  bilinear={rb:.4e}  (lin/bi={rl/rb:.1f}x)")
log(f"[bnd] c_x={cx:.4f} c_u={cu:.4f}  ||A||2={np.linalg.norm(M['A'],2):.4f}  spec.rad(A)={max(abs(np.linalg.eigvals(M['A']))):.4f}")

# ------------------------------------------------- CHECK 1: tube + CHECK 2: pred error
def roll_bi(z0, Us): return kc.roll_bi(M, z0, Us)
def roll_lin(z0, Us): return kc.roll_lin(M, z0, Us)

L = 40
r = np.random.default_rng(7)
eb = np.zeros(L+1); el = np.zeros(L+1); cnt = 0
mr, viol, tot, samp = 0.0, 0, 0, None
for _ in range(300):
    s0 = np.array([r.uniform(-1.5, 1.5), r.uniform(-1.5, 1.5), r.uniform(-np.pi, np.pi)])
    us = np.stack([r.uniform(-1.5, 1.5, L), r.uniform(-1.5, 1.5, L)], axis=1)
    ss = [s0.copy()]
    for u in us: ss.append(step(ss[-1], u))
    ss = np.array(ss); z0 = Psi(s0)
    zt = np.array([Psi(s) for s in ss]); zh = roll_bi(z0, us)
    # accuracy in original coords (x,y,theta)
    eb += np.linalg.norm((C @ zh.T).T - ss, axis=1)
    el += np.linalg.norm((C @ roll_lin(z0, us).T).T - ss, axis=1); cnt += 1
    # tube
    dl = np.linalg.norm(zt - zh, axis=1); e = kc.euclid_tube(M, zh, us, cx, cu)
    mr = max(mr, float(np.max(dl/np.maximum(e, 1e-12)))); viol += int(np.sum(dl > e + 1e-9)); tot += len(dl)
    if samp is None: samp = (dl, e)
eb /= cnt; el /= cnt
log(f"\n[1.tube] trajs={cnt}  max(||Delta||/e)={mr:.3f}  violations={viol}/{tot} ({100*viol/tot:.2f}%)")
log("[2.pred] open-loop error in (x,y,theta):")
for h in [5, 10, 20, 40]:
    log(f"[2.pred] H={h:2d}  bilinear={eb[h]:.4f}  linear={el[h]:.4f}  ratio(lin/bi)={el[h]/max(eb[h],1e-9):.1f}x")

# ----------------------------------------------- CHECK 3: MPPI parking to origin
def mppi(beta, steps=120, T=30, Ks=1024, lam=1.0, sig=np.array([0.6, 0.8]), seed=3):
    rr = np.random.default_rng(seed); s = np.array([1.5, 1.0, 0.5]); U = np.zeros((T, 2)); traj = [s.copy()]
    A, B0, B1, B2 = M["A"], M["B0"], M["Bs"][0], M["Bs"][1]
    nA = np.linalg.norm(A, 2); nB1 = np.linalg.norm(B1, 2); nB2 = np.linalg.norm(B2, 2)
    for _ in range(steps):
        z0 = Psi(s); eps = rr.normal(0, 1, size=(Ks, T, 2))*sig; V = U[None] + eps
        cost = np.zeros(Ks); Zb = np.tile(z0, (Ks, 1)); e = np.zeros(Ks)
        for k in range(T):
            uk = V[:, k, :]
            Zb = Zb@A.T + uk@B0.T + uk[:, 0:1]*(Zb@B1.T) + uk[:, 1:2]*(Zb@B2.T)
            sk = Zb@C.T
            cost += sk[:, 0]**2 + sk[:, 1]**2 + 0.3*sk[:, 2]**2 + 0.05*(uk**2).sum(1)
            if beta > 0:
                e = (nA + np.abs(uk[:, 0])*nB1 + np.abs(uk[:, 1])*nB2 + cx)*e + cx*np.linalg.norm(Zb, axis=1) + cu*np.linalg.norm(uk, axis=1)
                cost += beta*e
        sT = Zb@C.T; cost += 5.0*(sT[:, 0]**2 + sT[:, 1]**2 + sT[:, 2]**2)
        w = np.exp(-(cost-cost.min())/lam); w /= w.sum()
        U = U + np.einsum("k,ktd->td", w, eps)
        u0 = np.clip(U[0], [-1.5, -1.5], [1.5, 1.5]); s = step(s, u0); traj.append(s.copy())
        U = np.roll(U, -1, axis=0); U[-1] = 0.0
    return np.array(traj)
trj = mppi(beta=0.0)
log(f"\n[3.mppi] parking from (1.5,1.0,0.5): final (x,y,theta)=({trj[-1,0]:.3f},{trj[-1,1]:.3f},{trj[-1,2]:.3f})  "
    f"||pos||={np.linalg.norm(trj[-1,:2]):.4f}")

# ------------------------------------------------------------------------- plots
fig, AX = plt.subplots(1, 3, figsize=(15, 4.4))
dl, e = samp
AX[0].plot(dl, lw=2, label=r"$\|\Delta_k\|$"); AX[0].plot(e, "--", lw=2, label=r"$e_k$")
AX[0].set_yscale("log"); AX[0].set_title("unicycle: Check1 tube"); AX[0].set_xlabel("k"); AX[0].legend()
AX[1].plot(eb, lw=2, label="bilinear"); AX[1].plot(el, lw=2, label="linear")
AX[1].set_title("unicycle: Check2 pred err (x,y,theta)"); AX[1].set_xlabel("horizon"); AX[1].legend()
AX[2].plot(trj[:, 0], trj[:, 1], lw=2); AX[2].scatter([1.5], [1.0], c="C1", label="start")
AX[2].scatter([0], [0], c="k", marker="*", s=140, label="goal")
AX[2].set_title("unicycle: Check3 MPPI parking"); AX[2].set_xlabel("x"); AX[2].set_ylabel("y"); AX[2].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "tier1_unicycle.png"), dpi=110)

log("\n================= VERDICT =================")
log(f"[1] tube: {'PASS' if viol==0 else f'{100*viol/tot:.2f}% viol'} (max ratio {mr:.3f})")
log(f"[2] C1 bilinear<<linear: H=20 bi={eb[20]:.3f} vs lin={el[20]:.3f}  ({el[20]/max(eb[20],1e-9):.1f}x)  "
    f"{'PASS' if el[20]>2*eb[20] else 'weak'}")
log(f"[3] MPPI parking: {'PASS' if np.linalg.norm(trj[-1,:2])<0.3 else 'CHECK'}")
log(f"\noutputs -> {OUT}/tier1_unicycle.png")
with open(os.path.join(OUT, "tier1_summary.txt"), "w") as fh:
    fh.write("\n".join(_log) + "\n")
