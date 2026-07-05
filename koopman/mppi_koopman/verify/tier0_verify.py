"""
Tier-0 numerical verification for RB-KMPPI (notes/derivation.md, notes/formulation_AB.md).

Two systems (same degree-3 polynomial lifting, Psi(0)=0, C=[I2|0]):
  A) vanderpol      : control-affine with CONSTANT input matrix G=[0;1]
                      -> lifted bilinear terms are MILD (weak bilinear advantage expected).
  B) pendulum_sdg   : torque enters as cos(x1)*u  (STATE-DEPENDENT gain G(x)=[0;cos x1])
                      -> linear EDMDc must approximate cos(x1)*u by a constant Bu (fails);
                         bilinear captures it via u*(x1^2 ...) terms -> STRONG advantage expected.

Checks per system:
  [1] Theorem 1 (tube):  e_k >= ||Delta_k|| on test trajs (in-region + out-of-region stress = R1).
  [2] C1 (accuracy):     open-loop prediction error in x, bilinear vs linear, vs horizon.
  [3] (aux) closed-loop: MPPI regulation with bilinear surrogate (+ optional tube penalty).
Outputs: ./out/summary.txt , ./out/tier0_verify.png
"""
import os
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
_log = []
def log(m=""):
    print(m); _log.append(str(m))

DT = 0.05
def f_ct(x, u, sys):
    x1, x2 = x[0], x[1]
    if sys == "vanderpol":
        return np.array([x2, 1.0*(1.0 - x1*x1)*x2 - x1 + u])           # G = [0;1] constant
    else:  # pendulum_sdg
        return np.array([x2, -np.sin(x1) - 0.2*x2 + np.cos(x1)*u])     # G(x) = [0; cos x1]
def step(x, u, sys):
    k1 = f_ct(x, u, sys); k2 = f_ct(x + 0.5*DT*k1, u, sys)
    k3 = f_ct(x + 0.5*DT*k2, u, sys); k4 = f_ct(x + DT*k3, u, sys)
    return x + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def Psi(x):
    x1, x2 = x[0], x[1]
    return np.array([x1, x2, x1*x1, x1*x2, x2*x2, x1**3, x1*x1*x2, x1*x2*x2, x2**3])
N = 9
C = np.zeros((2, N)); C[0, 0] = 1.0; C[1, 1] = 1.0

def make_traj(x0, us, sys):
    xs = [x0.copy()]
    for u in us: xs.append(step(xs[-1], u, sys))
    return np.array(xs)

def gather(sys, n_traj, L, u_amp, x_amp, seed):
    r = np.random.default_rng(seed); X, U, Y = [], [], []
    for _ in range(n_traj):
        x0 = r.uniform(-x_amp, x_amp, size=2); us = r.uniform(-u_amp, u_amp, size=L)
        xs = make_traj(x0, us, sys)
        if np.max(np.abs(xs)) > 6.0: continue
        for k in range(L):
            X.append(Psi(xs[k])); U.append(us[k]); Y.append(Psi(xs[k+1]))
    return np.array(X).T, np.array(U)[None, :], np.array(Y).T

def rmse(P, T): return float(np.sqrt(np.mean(np.sum((P-T)**2, axis=0))))

def identify(sys):
    Zx, Uu, Zy = gather(sys, 300, 40, 2.0, 2.0, seed=1)
    Th_lin = Zy @ np.linalg.pinv(np.vstack([Zx, Uu]))
    A_lin, B_lin = Th_lin[:, :N], Th_lin[:, N:N+1]
    Th_bi = Zy @ np.linalg.pinv(np.vstack([Zx, Uu, Uu*Zx]))
    A, B0, B1 = Th_bi[:, :N], Th_bi[:, N:N+1], Th_bi[:, N+1:]
    rl = rmse(A_lin @ Zx + B_lin @ Uu, Zy)
    rb = rmse(A @ Zx + B0 @ Uu + Uu*(B1 @ Zx), Zy)
    # proportional bound c_x,c_u via LP (full coverage on train residuals)
    Rres = Zy - (A @ Zx + B0 @ Uu + Uu*(B1 @ Zx))
    rn, zn, un = np.linalg.norm(Rres, axis=0), np.linalg.norm(Zx, axis=0), np.abs(Uu[0])
    res = linprog([1.0, 1.0], A_ub=-np.vstack([zn, un]).T, b_ub=-rn,
                  bounds=[(0, None), (0, None)], method="highs")
    cx, cu = res.x
    return dict(A=A, B0=B0, B1=B1, A_lin=A_lin, B_lin=B_lin, cx=cx, cu=cu,
                nA=np.linalg.norm(A, 2), nB1=np.linalg.norm(B1, 2), rl=rl, rb=rb, D=Zx.shape[1])

def roll_bi(M, z0, us):
    z = z0.copy(); Z = [z]
    for u in us: z = M["A"] @ z + M["B0"][:, 0]*u + u*(M["B1"] @ z); Z.append(z)
    return np.array(Z)
def roll_lin(M, z0, us):
    z = z0.copy(); Z = [z]
    for u in us: z = M["A_lin"] @ z + M["B_lin"][:, 0]*u; Z.append(z)
    return np.array(Z)

def tube(M, zhat, us):
    e = 0.0; E = [0.0]
    for k, u in enumerate(us):
        e = (M["nA"] + abs(u)*M["nB1"] + M["cx"])*e + M["cx"]*np.linalg.norm(zhat[k]) + M["cu"]*abs(u)
        E.append(e)
    return np.array(E)

def tube_check(M, sys, n, L, u_amp, x_amp, seed):
    r = np.random.default_rng(seed); mr, viol, tot, samp = 0.0, 0, 0, None
    for _ in range(n):
        x0 = r.uniform(-x_amp, x_amp, size=2); us = r.uniform(-u_amp, u_amp, size=L)
        xs = make_traj(x0, us, sys)
        if np.max(np.abs(xs)) > 6.0: continue
        zt = np.array([Psi(x) for x in xs]); zh = roll_bi(M, zt[0], us)
        dl = np.linalg.norm(zt - zh, axis=1); e = tube(M, zh, us)
        mr = max(mr, float(np.max(dl/np.maximum(e, 1e-12))))
        viol += int(np.sum(dl > e + 1e-9)); tot += len(dl)
        if samp is None: samp = (dl, e)
    return mr, viol, tot, samp

def pred_curve(M, sys, n, L, u_amp, x_amp, seed):
    r = np.random.default_rng(seed); eb, el, c = np.zeros(L+1), np.zeros(L+1), 0
    for _ in range(n):
        x0 = r.uniform(-x_amp, x_amp, size=2); us = r.uniform(-u_amp, u_amp, size=L)
        xs = make_traj(x0, us, sys)
        if np.max(np.abs(xs)) > 6.0: continue
        z0 = Psi(x0)
        eb += np.linalg.norm((C @ roll_bi(M, z0, us).T).T - xs, axis=1)
        el += np.linalg.norm((C @ roll_lin(M, z0, us).T).T - xs, axis=1); c += 1
    return eb/c, el/c, c

def mppi(M, sys, beta, steps=100, T=25, Ks=512, lam=1.0, sig=0.6, seed=3):
    r = np.random.default_rng(seed); x = np.array([1.6, 1.6]); U = np.zeros(T); traj = [x.copy()]
    for _ in range(steps):
        z0 = Psi(x); eps = r.normal(0, sig, size=(Ks, T)); V = U[None, :] + eps
        cost = np.zeros(Ks); Zb = np.tile(z0, (Ks, 1)); e = np.zeros(Ks)
        for k in range(T):
            uk = V[:, k]
            Zb = (Zb @ M["A"].T) + np.outer(uk, M["B0"][:, 0]) + uk[:, None]*(Zb @ M["B1"].T)
            xk = Zb @ C.T; cost += xk[:, 0]**2 + xk[:, 1]**2 + 0.05*uk**2
            if beta > 0:
                e = (M["nA"] + np.abs(uk)*M["nB1"] + M["cx"])*e + M["cx"]*np.linalg.norm(Zb, axis=1) + M["cu"]*np.abs(uk)
                cost += beta*e
        xT = Zb @ C.T; cost += 10.0*(xT[:, 0]**2 + xT[:, 1]**2)
        w = np.exp(-(cost - cost.min())/lam); w /= w.sum()
        U = U + w @ eps; x = step(x, float(np.clip(U[0], -3, 3)), sys); traj.append(x.copy())
        U = np.roll(U, -1); U[-1] = 0.0
    return np.array(traj)

# ------------------------------------------------------------------------------- run
SYS = ["vanderpol", "pendulum_sdg"]
fig, AX = plt.subplots(2, 3, figsize=(15, 8.4))
results = {}
for row, sys in enumerate(SYS):
    log(f"\n################## SYSTEM: {sys} ##################")
    M = identify(sys)
    log(f"[fit ] D={M['D']}  1-step lifted RMSE  linear={M['rl']:.4e}  bilinear={M['rb']:.4e}  "
        f"(lin/bi={M['rl']/M['rb']:.2f}x)")
    log(f"[bnd ] c_x={M['cx']:.4f} c_u={M['cu']:.4f}  ||A||2={M['nA']:.4f} ||B1||2={M['nB1']:.4f}")

    mr_in, v_in, n_in, samp = tube_check(M, sys, 200, 40, 2.0, 2.0, seed=7)
    mr_out, v_out, n_out, _ = tube_check(M, sys, 200, 40, 3.0, 3.0, seed=8)
    log(f"[1.tube] in-region : max(||D||/e)={mr_in:.3f}  viol={v_in}/{n_in} ({100*v_in/n_in:.2f}%)")
    log(f"[1.tube] out-region: max(||D||/e)={mr_out:.3f}  viol={v_out}/{n_out} ({100*v_out/n_out:.2f}%)  <- R1 stress")

    eb, el, c = pred_curve(M, sys, 200, 40, 2.0, 2.0, seed=9)
    log(f"[2.pred] test trajs={c}")
    for h in [5, 10, 20, 40]:
        log(f"[2.pred] H={h:2d}  bilinear={eb[h]:.4f}  linear={el[h]:.4f}  ratio(lin/bi)={el[h]/max(eb[h],1e-9):.2f}x")

    betas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    finals = [float(np.linalg.norm(mppi(M, sys, b)[-1])) for b in betas]
    bi = int(np.argmin(finals))
    log(f"[3.mppi] final ||x|| vs beta: " + "  ".join(f"{b}->{f:.3f}" for b, f in zip(betas, finals)))
    log(f"[3.mppi] best beta*={betas[bi]} (||x||={finals[bi]:.4f});  beta=0 (pure KMPPI)={finals[0]:.4f}")
    results[sys] = dict(v_in=v_in, v_out=v_out, n_out=n_out, eb=eb, el=el, betas=betas, finals=finals, bi=bi)

    dl, e = samp
    AX[row, 0].plot(dl, lw=2, label=r"$\|\Delta_k\|$"); AX[row, 0].plot(e, "--", lw=2, label=r"$e_k$")
    AX[row, 0].set_yscale("log"); AX[row, 0].set_title(f"{sys}: Check1 tube"); AX[row, 0].legend()
    AX[row, 1].plot(eb, lw=2, label="bilinear"); AX[row, 1].plot(el, lw=2, label="linear")
    AX[row, 1].set_title(f"{sys}: Check2 pred err"); AX[row, 1].set_xlabel("horizon"); AX[row, 1].legend()
    AX[row, 2].plot(betas, finals, "o-", lw=2); AX[row, 2].axvline(betas[bi], ls=":", c="g")
    AX[row, 2].set_title(f"{sys}: Check3 MPPI beta-sweep"); AX[row, 2].set_xlabel(r"$\beta$"); AX[row, 2].set_ylabel(r"final $\|x\|$")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "tier0_verify.png"), dpi=110)

log("\n================= VERDICT =================")
for sys in SYS:
    R = results[sys]
    log(f"[{sys}]")
    log(f"   [1] tube in-region : {'PASS' if R['v_in']==0 else 'FAIL'} (viol={R['v_in']});  "
        f"out-region viol={100*R['v_out']/R['n_out']:.2f}% (R1 caveat -> expect >0)")
    log(f"   [2] bilinear<linear: H=40 bi={R['eb'][40]:.3f} vs lin={R['el'][40]:.3f}  "
        f"({'PASS' if R['el'][40]>R['eb'][40] else 'FAIL'}); best ratio={max(R['el']/np.maximum(R['eb'],1e-9)):.2f}x")
    log(f"   [3] MPPI stabilized: {'PASS' if R['finals'][R['bi']]<0.3 else 'CHECK'} "
        f"(best beta*={R['betas'][R['bi']]}, ||x||={R['finals'][R['bi']]:.4f}); "
        f"too-large beta over-conservative -> see sweep")
log(f"\noutputs -> {OUT}")
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write("\n".join(_log) + "\n")
