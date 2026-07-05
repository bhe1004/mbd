"""Can we make the error tube CONTRACTIVE (coef<1, Theorem-1 ultimate bound holds) on the unicycle,
via stability-constrained identification (eigenvalue clipping = SafEDMD stand-in) + the P-LMI weighted tube?
Time-boxed check. Hypothesis: NO, because the unicycle is a kinematic integrator (lifted A has eigvals~1),
so the clipping error inflates c_x ~ (1-target) and coef = rho + kappa_P*c_x stays >= 1."""
import os, itertools, numpy as np, cvxpy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import koopman_core as kc
FIGS = "/figs"; DT = kc.DT

def f_ct(s, u): return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
def step(s, u):
    k1 = f_ct(s, u); k2 = f_ct(s+0.5*DT*k1, u); k3 = f_ct(s+0.5*DT*k2, u); k4 = f_ct(s+DT*k3, u)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi(s): return np.array([s[0], s[1], s[2], np.sin(s[2]), np.cos(s[2])-1.0])
def gather(n, L, seed, ubox=1.5, pbox=2.0):
    r = np.random.default_rng(seed); X, U, Y = [], [], []
    for _ in range(n):
        s0 = np.array([r.uniform(-pbox, pbox), r.uniform(-pbox, pbox), r.uniform(-np.pi, np.pi)])
        us = np.stack([r.uniform(-ubox, ubox, L), r.uniform(-ubox, ubox, L)], 1)
        ss = [s0.copy()]
        for u in us: ss.append(step(ss[-1], u))
        ss = np.array(ss)
        for k in range(L):
            X.append(Psi(ss[k])); U.append(us[k]); Y.append(Psi(ss[k+1]))
    return np.array(X).T, np.array(U).T, np.array(Y).T

def solve_plmi(M, u_box, rho, kap2=1e6):
    N, m = M["N"], M["m"]; verts = list(itertools.product(*[(-u_box, u_box)]*m))
    P = cp.Variable((N, N), PSD=True); t = cp.Variable(nonneg=True)
    cons = [P >> np.eye(N), P << t*np.eye(N), t <= kap2]
    for v in verts:
        Mv = M["A"] + sum(v[i]*M["Bs"][i] for i in range(m))
        cons.append(cp.bmat([[rho*rho*P, Mv.T @ P], [P @ Mv, P]]) >> 0)
    try: cp.Problem(cp.Minimize(t), cons).solve(solver=cp.CLARABEL)
    except Exception: return None
    return None if P.value is None else np.array(P.value)
def find_best(M, u_box, cx):
    sr = max(abs(np.linalg.eigvals(M["A"]))); best = None
    for rho in np.linspace(min(sr+0.02, 0.97), 0.995, 14):
        P = solve_plmi(M, u_box, rho)
        if P is None: continue
        ev = np.linalg.eigvalsh(P)
        if ev.min() <= 0: continue
        kP = float(np.sqrt(ev.max()/ev.min())); coef = rho + kP*cx
        if best is None or coef < best[2]: best = (rho, P, coef, kP)
    return best
def refit_B(A_s, Zx, Uu, Zy):
    N, m = Zx.shape[0], Uu.shape[0]
    Phiu = np.vstack([Uu] + [Uu[i:i+1, :]*Zx for i in range(m)])
    Th = (Zy - A_s @ Zx) @ np.linalg.pinv(Phiu)
    return Th[:, :m], [Th[:, m+i*N: m+(i+1)*N] for i in range(m)]
def stabilize(M0, Zx, Uu, Zy, target):
    A = M0["A"]; w, V = np.linalg.eig(A)
    A_s = np.real(V @ np.diag(w*np.where(np.abs(w) > target, target/np.abs(w), 1.0)) @ np.linalg.inv(V))
    B0, Bs = refit_B(A_s, Zx, Uu, Zy)
    Ms = dict(N=M0["N"], m=M0["m"], A=A_s, B0=B0, Bs=Bs, A_lin=M0["A_lin"], B_lin=M0["B_lin"])
    return Ms, max(abs(np.linalg.eigvals(A_s)))
def ptube(M, zhat, Us, cx, cu, rho, P):
    lmax = float(np.max(np.linalg.eigvalsh(P))); lmin = float(np.min(np.linalg.eigvalsh(P)))
    kP = np.sqrt(lmax/lmin); coef = rho + kP*cx; e = 0.0; E = [0.0]
    for k, u in enumerate(np.atleast_2d(Us)):
        e = coef*e + np.sqrt(lmax)*(cx*np.linalg.norm(zhat[k]) + cu*np.linalg.norm(u)); E.append(e)
    return np.array(E)/np.sqrt(lmin), kP, coef

Zx, Uu, Zy = gather(300, 40, 1)
M0 = kc.identify(Zx, Uu, Zy)
cx0, cu0 = kc.fit_bound(M0, Zx, Uu, Zy)
print(f"[vanilla] ||A||2={np.linalg.norm(M0['A'],2):.3f} spec.rad(A)={max(abs(np.linalg.eigvals(M0['A']))):.4f} c_x={cx0:.3e} c_u={cu0:.3e}")
print(f"  eig(A) magnitudes = {sorted(np.round(np.abs(np.linalg.eigvals(M0['A'])),3))}  <-- integrator modes near 1?")

best_overall = None
for UBOX in (0.3, 0.5):
    for target in (0.99, 0.97, 0.95):
        Ms, srs = stabilize(M0, Zx, Uu, Zy, target)
        cxs, cus = kc.fit_bound(Ms, Zx, Uu, Zy)
        b = find_best(Ms, UBOX, cxs)
        tag = f"UBOX={UBOX} target={target}: spec.rad(A_s)={srs:.3f} c_x={cxs:.3e}"
        if b is None:
            print(f"[{tag}] P-LMI infeasible")
        else:
            rho, P, coef, kP = b
            print(f"[{tag}] rho*={rho:.3f} kappa_P={kP:.1f} coef={coef:.4f} {'CONTRACTIVE' if coef<1 else '>=1 (not contractive)'}")
            if coef < 1 and (best_overall is None or coef < best_overall[0]):
                best_overall = (coef, UBOX, target, Ms, cxs, cus, rho, P)

if best_overall is not None:
    coef, UBOX, target, Ms, cxs, cus, rho, P = best_overall
    print(f"\n*** CONTRACTIVE tube achieved: coef={coef:.4f} (UBOX={UBOX}, target={target}) ***")
    r = np.random.default_rng(7); L = 40
    s0 = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-0.5, 0.5)])
    us = np.stack([r.uniform(-UBOX, UBOX, L), r.uniform(-UBOX, UBOX, L)], 1)
    ss = [s0.copy()]
    for u in us: ss.append(step(ss[-1], u))
    ss = np.array(ss); zt = np.array([Psi(s) for s in ss])
    zhs = kc.roll_bi(Ms, zt[0], us); dls = np.linalg.norm(zt - zhs, axis=1)
    eEs = kc.euclid_tube(Ms, zhs, us, cxs, cus); eP, kP, coef = ptube(Ms, zhs, us, cxs, cus, rho, P)
    print(f"  valid(eP>=true)={bool(np.all(eP+1e-9>=dls))}  eP_max={eP.max():.3e}  eEuclid_max={eEs.max():.3e}")
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.plot(dls, color="#1f77b4", lw=2.0, label=r"true lifted error $\|z_k-\hat z_k\|$")
    ax.plot(eEs, "--", color="#999999", lw=1.8, label="Euclidean tube (non-contractive)")
    ax.plot(eP, "-.", color="#d62728", lw=2.4, label=r"P-LMI tube $e_k^P$ (contractive)")
    ax.set_yscale("log"); ax.set_xlabel(r"rollout step $k$"); ax.set_ylabel("lifted-space error")
    ax.grid(True, alpha=0.35); ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tube_contractive.png"), dpi=200)
    print("saved /figs/fig_tube_contractive.png")
else:
    print("\n*** NO contractive tube: every config gives coef>=1 (integrator obstruction confirmed). Keep proxy framing. ***")
