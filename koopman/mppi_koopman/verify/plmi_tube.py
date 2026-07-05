"""
Verify section 6 (P-LMI weighted-norm tube) vs section 1 (Euclidean tube).

tier0 finding: Euclidean tube e_k explodes (||A||_2>1 -> coeff mbar+c_x>1).
Section 6 replaces ||M|| by the true contraction rate rho via M(v)^T P M(v) <= rho^2 P.

EXTRA FINDING (this script): vanilla least-squares EDMDc yields spectrally UNSTABLE
lifted A (spec.radius>1) even for a physically stable system -> the guarantee chain
needs stability-constrained identification (SafEDMD).  We emulate that with ridge
regularization (shrinks A), then show section 6 engages and bounds the tube.
"""
import os, itertools
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koopman_core as kc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
_log = []
def log(m=""):
    print(m); _log.append(str(m))

def solve_plmi(M, u_box, rho, kap2=1e6):
    """feasibility + minimize conditioning: min t s.t. I <= P <= t I, Schur LMIs at vertices."""
    N, m = M["N"], M["m"]
    verts = list(itertools.product(*[(-u_box, u_box)] * m))
    P = cp.Variable((N, N), PSD=True); t = cp.Variable(nonneg=True)
    cons = [P >> np.eye(N), P << t*np.eye(N), t <= kap2]
    for v in verts:
        Mv = M["A"] + sum(v[i]*M["Bs"][i] for i in range(m))
        cons.append(cp.bmat([[rho*rho*P, Mv.T @ P], [P @ Mv, P]]) >> 0)
    try:
        cp.Problem(cp.Minimize(t), cons).solve(solver=cp.CLARABEL)
    except Exception:
        return None
    return None if P.value is None else np.array(P.value)

def find_best(M, u_box, cx):
    """choose rho to MINIMIZE the tube coefficient rho + kappa_P(rho)*c_x (not just min rho:
    smaller rho forces ill-conditioned P -> larger kappa_P -> larger coef)."""
    sr = max(abs(np.linalg.eigvals(M["A"])))
    best = None
    for rho in np.linspace(min(sr + 0.02, 0.97), 0.995, 14):
        P = solve_plmi(M, u_box, rho)
        if P is None:
            continue
        ev = np.linalg.eigvalsh(P)
        if ev.min() <= 0:
            continue
        kP = float(np.sqrt(ev.max()/ev.min())); coef = rho + kP*cx
        if best is None or coef < best[2]:
            best = (rho, P, coef, kP)
    return best

def refit_B(A_s, Zx, Uu, Zy):
    N, m = Zx.shape[0], Uu.shape[0]
    Phiu = np.vstack([Uu] + [Uu[i:i+1, :]*Zx for i in range(m)])
    Th = (Zy - A_s @ Zx) @ np.linalg.pinv(Phiu)
    return Th[:, :m], [Th[:, m+i*N: m+(i+1)*N] for i in range(m)]

def stabilize(M0, Zx, Uu, Zy, target=0.99):
    """eigenvalue clipping (accuracy-preserving): nudge only |lambda|>target below target,
    keep stable modes intact, then refit B with A fixed -> small c_x.  Stand-in for SafEDMD."""
    A = M0["A"]; w, V = np.linalg.eig(A)
    A_s = np.real(V @ np.diag(w*np.where(np.abs(w) > target, target/np.abs(w), 1.0)) @ np.linalg.inv(V))
    B0, Bs = refit_B(A_s, Zx, Uu, Zy)
    Ms = dict(N=M0["N"], m=M0["m"], A=A_s, B0=B0, Bs=Bs, A_lin=M0["A_lin"], B_lin=M0["B_lin"])
    sr = max(abs(np.linalg.eigvals(A_s)))
    rmse = float(np.sqrt(np.mean(np.sum((kc.predict_bi(Ms, Zx, Uu) - Zy)**2, axis=0))))
    return Ms, sr, rmse

def ptube(M, zhat, Us, cx, cu, rho, P):
    lmax = float(np.max(np.linalg.eigvalsh(P))); lmin = float(np.min(np.linalg.eigvalsh(P)))
    kP = np.sqrt(lmax/lmin); coef = rho + kP*cx
    e = 0.0; E = [0.0]
    for k, u in enumerate(np.atleast_2d(Us)):
        e = coef*e + np.sqrt(lmax)*(cx*np.linalg.norm(zhat[k]) + cu*np.linalg.norm(u)); E.append(e)
    return np.array(E)/np.sqrt(lmin), kP, coef

UBOX = 0.5            # operating input range for robust contraction (near-equilibrium regime)
def test_traj(sys):
    r = np.random.default_rng(11)
    x0 = r.uniform(-1.0, 1.0, size=2); us = r.uniform(-UBOX, UBOX, size=40)
    xs = [x0.copy()]
    for u in us: xs.append(kc.step_2d(xs[-1], u, sys))
    return np.array(xs), us[:, None]

def synthetic_demo(ax):
    """Controlled synthetic bilinear system with KNOWN small c_x: isolates the sec6
    MECHANISM. A is non-normal & Schur-stable (spec.rad=0.85, ||A||_2~3>1) so the
    Euclidean tube explodes but the P-LMI tube stays bounded & tight."""
    rng = np.random.default_rng(0)
    A = np.array([[0.85, 2.0], [0.0, 0.85]]); B0 = np.array([[0.1], [0.1]]); B1 = 0.1*np.eye(2)
    M = dict(N=2, m=1, A=A, B0=B0, Bs=[B1], A_lin=A, B_lin=B0)
    cx, cu = 0.001, 0.005
    us = rng.uniform(-UBOX, UBOX, 40); zc = rng.normal(size=2); zt = [zc.copy()]
    for u in us:                                        # true rollout w/ worst-case residual
        base = A@zc + B0[:, 0]*u + u*(B1@zc)
        d = rng.normal(size=2); d /= np.linalg.norm(d)
        zc = base + (cx*np.linalg.norm(zc) + cu*abs(u))*d; zt.append(zc.copy())
    zt = np.array(zt); zh = kc.roll_bi(M, zt[0], us[:, None]); dl = np.linalg.norm(zt - zh, axis=1)
    eE = kc.euclid_tube(M, zh, us[:, None], cx, cu)
    rho, P, coef, kP = find_best(M, UBOX, cx); eP, kP, coef = ptube(M, zh, us[:, None], cx, cu, rho, P)
    log(f"\n################## synthetic (known small c_x) ##################")
    log(f"[synth] ||A||2={np.linalg.norm(A,2):.2f} spec.rad={max(abs(np.linalg.eigvals(A))):.2f}  c_x={cx} c_u={cu}")
    log(f"[synth] P-LMI rho*={rho:.4f} kappa_P={kP:.1f} coef={coef:.4f} (<1 -> CONTRACTIVE)  valid={bool(np.all(eP+1e-9>=dl))}")
    log(f"[synth] e_max  Euclid={eE.max():.3e}  P-LMI={eP.max():.3e}  -> sec6 tightens {eE.max()/eP.max():.2e}x")
    ax.plot(dl, "k", lw=1.5, label=r"$\|\Delta_k\|$"); ax.plot(eE, "C1--", lw=2, label=r"Euclid $e_k$")
    ax.plot(eP, "C2-", lw=2.5, label=r"$e_k^P$ sec6"); ax.set_yscale("log")
    ax.set_title(f"synthetic (c_x={cx}): sec6 WORKS\ncoef={coef:.2f}<1, tighten {eE.max()/eP.max():.0e}x"); ax.set_xlabel("k"); ax.legend(fontsize=8)

fig, AX = plt.subplots(1, 3, figsize=(16.5, 4.8))
synthetic_demo(AX[0])
for col0, sys in enumerate(["pendulum_sdg", "vanderpol"]):
    col = col0 + 1
    log(f"\n################## {sys} ##################")
    Zx, Uu, Zy = kc.gather_2d(sys, 300, 40, 2.0, 2.0, seed=1)
    M0 = kc.identify(Zx, Uu, Zy)
    sr0 = max(abs(np.linalg.eigvals(M0["A"])))
    cx0, cu0 = kc.fit_bound(M0, Zx, Uu, Zy)
    log(f"[vanilla] ||A||2={np.linalg.norm(M0['A'],2):.3f} spec.rad(A)={sr0:.4f} "
        f"-> P-LMI {'feasible' if sr0<1 else 'INFEASIBLE (unstable lifted A)'}")

    xs, Us = test_traj(sys)
    zt = np.array([kc.Psi_2d(x) for x in xs]); zh0 = kc.roll_bi(M0, zt[0], Us)
    dl0 = np.linalg.norm(zt - zh0, axis=1)
    eE0 = kc.euclid_tube(M0, zh0, Us, cx0, cu0)

    # stability-constrained identification (eigenvalue clipping, accuracy-preserving)
    Ms, srs, rmses = stabilize(M0, Zx, Uu, Zy, target=0.95)
    rmse0 = float(np.sqrt(np.mean(np.sum((kc.predict_bi(M0, Zx, Uu) - Zy)**2, axis=0))))
    srv = max(max(abs(np.linalg.eigvals(Ms["A"] + s*Ms["Bs"][0]))) for s in (-UBOX, UBOX))
    log(f"[eigclip] spec.rad(A)={srs:.4f} ||A||2={np.linalg.norm(Ms['A'],2):.3f}  "
        f"spec.rad(M(+-{UBOX}))={srv:.4f}  (1-step RMSE {rmse0:.3e}->{rmses:.3e})")
    cxs, cus = kc.fit_bound(Ms, Zx, Uu, Zy)
    bestS = find_best(Ms, UBOX, cxs)
    rho, P = (bestS[0], bestS[1]) if bestS is not None else (None, None)
    zhs = kc.roll_bi(Ms, zt[0], Us); dls = np.linalg.norm(zt - zhs, axis=1)
    eEs = kc.euclid_tube(Ms, zhs, Us, cxs, cus)
    if P is not None:
        eP, kP, coef = ptube(Ms, zhs, Us, cxs, cus, rho, P)
        valid = bool(np.all(eP + 1e-9 >= dls))
        log(f"[sec6  ] P-LMI rho*={rho:.4f} kappa_P={kP:.1f} coef(rho+kP*cx)={coef:.4f}  "
            f"valid(eP>=||Delta||)={valid}")
        log(f"[result] e_max  Euclid(vanilla)={eE0.max():.2e}  Euclid(stable)={eEs.max():.2e}  "
            f"P-LMI(sec6)={eP.max():.3f}  ->  sec6 tightens {eEs.max()/eP.max():.1e}x vs Euclid-stable")
    else:
        log(f"[sec6  ] P-LMI infeasible even with stable A over box +-{UBOX} "
            f"(c_x={cxs:.3f} too large / box too wide -> needs SafEDMD-grade ID).")
        eP = None

    AX[col].plot(dl0, "0.6", lw=1.5, label=r"$\|\Delta_k\|$ vanilla")
    AX[col].plot(eE0, "C1--", lw=1.5, label=r"Euclid $e_k$ vanilla")
    AX[col].plot(eEs, "C3:", lw=1.5, label=r"Euclid $e_k$ stable-A")
    if eP is not None:
        AX[col].plot(eP, "C2-", lw=2.5, label=r"$e_k^P$ sec6 (stable-A)")
        AX[col].plot(dls, "k", lw=1.2, label=r"$\|\Delta_k\|$ stable-A")
    AX[col].set_yscale("log"); AX[col].set_xlabel("k"); AX[col].legend(fontsize=7.5)
    AX[col].set_title(f"{sys}: sec1 Euclid tube vs sec6 P-LMI tube")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "plmi_tube.png"), dpi=110)
log(f"\noutputs -> {OUT}/plmi_tube.png")
with open(os.path.join(OUT, "plmi_summary.txt"), "w") as fh:
    fh.write("\n".join(_log) + "\n")
