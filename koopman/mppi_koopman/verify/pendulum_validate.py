"""
Implementation-correctness audit for the pendulum Koopman models (no control, just the model).
 [A] rollout consistency : batched MPPI-style bilinear rollout == sequential roll_bi == predict_bi
 [B] 1-step generalization: held-out TEST RMSE (not training) for linear vs bilinear
 [C] multi-step open-loop : prediction error vs horizon, bilinear vs linear, in broad & near-upright regimes
Tells us whether 'bilinear more accurate' is real (and not a training-RMSE artifact / rollout bug).
"""
import numpy as np
import koopman_core as kc

GL, B, ML2, DT, UMAX = 10.0, 0.1, 1.0, 0.02, 8.0
def f(s, u): return np.array([s[1], GL*np.sin(s[0]) - B*s[1] + u/ML2])
def step(s, u):
    k1 = f(s, u); k2 = f(s+0.5*DT*k1, u); k3 = f(s+0.5*DT*k2, u); k4 = f(s+DT*k3, u)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi(s):
    th, dth = s[0], s[1]
    return np.array([np.sin(th), np.cos(th)-1.0, dth, np.sin(th)*dth, np.cos(th)*dth, dth*dth])
N = 6; C = np.zeros((3, N)); C[0, 0] = C[1, 1] = C[2, 2] = 1.0
A_ = None  # set after identify

def gather(D, useed):
    r = np.random.default_rng(useed)
    TH = r.uniform(-1.4, 1.4, D); DTH = r.uniform(-5, 5, D); US = r.uniform(-UMAX, UMAX, D)
    X = np.array([Psi(np.array([TH[i], DTH[i]])) for i in range(D)]).T
    Y = np.array([Psi(step(np.array([TH[i], DTH[i]]), US[i])) for i in range(D)]).T
    return X, US[None, :], Y

Zx, Uu, Zy = gather(20000, 1)
M = kc.identify(Zx, Uu, Zy)
A, B0, B1 = M["A"], M["B0"], M["Bs"][0]; Al, Bl = M["A_lin"], M["B_lin"]

# ---------- [A] rollout consistency ----------
rng = np.random.default_rng(5)
s0 = np.array([0.3, 1.0]); us = rng.uniform(-UMAX, UMAX, 30)
z_seq = kc.roll_bi(M, Psi(s0), us[:, None])                # sequential (core)
# batched MPPI-style with Ks=3 identical samples
Ks = 3; Zb = np.tile(Psi(s0), (Ks, 1)); Z_batched = [Zb.copy()]
for u in us:
    uk = np.full(Ks, u)
    Zb = Zb@A.T + np.outer(uk, B0[:, 0]) + uk[:, None]*(Zb@B1.T)
    Z_batched.append(Zb.copy())
Z_batched = np.array(Z_batched)[:, 0, :]                   # take sample 0
# predict_bi composition (manual one-step)
z = Psi(s0).copy(); Z_pred = [z.copy()]
for u in us:
    z = A@z + B0[:, 0]*u + u*(B1@z); Z_pred.append(z.copy())
Z_pred = np.array(Z_pred)
e_sb = np.max(np.abs(z_seq - Z_batched)); e_sp = np.max(np.abs(z_seq - Z_pred))
print("[A] rollout consistency (max abs diff):")
print(f"    sequential vs batched-MPPI = {e_sb:.2e}   sequential vs predict_bi = {e_sp:.2e}   "
      f"-> {'OK' if max(e_sb,e_sp) < 1e-10 else 'MISMATCH!'}")

# ---------- [B] 1-step generalization (held-out TEST set) ----------
ZxT, UuT, ZyT = gather(20000, 999)                         # fresh seed = test
def rmse(P, T): return float(np.sqrt(np.mean(np.sum((P-T)**2, axis=0))))
rl_tr = rmse(Al@Zx + Bl@Uu, Zy);  rb_tr = rmse(kc.predict_bi(M, Zx, Uu), Zy)
rl_te = rmse(Al@ZxT + Bl@UuT, ZyT); rb_te = rmse(kc.predict_bi(M, ZxT, UuT), ZyT)
print("\n[B] 1-step RMSE (lifted):")
print(f"    TRAIN  linear={rl_tr:.4e}  bilinear={rb_tr:.4e}  (lin/bi={rl_tr/rb_tr:.2f}x)")
print(f"    TEST   linear={rl_te:.4e}  bilinear={rb_te:.4e}  (lin/bi={rl_te/rb_te:.2f}x)  <- generalization")

# ---------- [C] multi-step open-loop prediction error vs horizon ----------
def openloop(regime, ntraj, H, seed):
    r = np.random.default_rng(seed); eb = np.zeros(H+1); el = np.zeros(H+1); cnt = 0
    for _ in range(ntraj):
        if regime == "broad":
            s = np.array([r.uniform(-1.3, 1.3), r.uniform(-4, 4)]); us = r.uniform(-UMAX, UMAX, H)
        else:  # near-upright (balancing operating region)
            s = np.array([r.uniform(-0.5, 0.5), r.uniform(-2, 2)]); us = r.uniform(-3, 3, H)
        ss = [s.copy()]
        for u in us: ss.append(step(ss[-1], u))
        ss = np.array(ss)
        if np.max(np.abs(ss[:, 0])) > 2.5: continue
        z0 = Psi(s)
        xb = (C @ kc.roll_bi(M, z0, us[:, None]).T).T       # (sinθ,cosθ-1,θ̇) predicted
        xl = (C @ kc.roll_lin(M, z0, us[:, None]).T).T
        xt = np.array([[np.sin(p[0]), np.cos(p[0])-1.0, p[1]] for p in ss])
        eb += np.linalg.norm(xb - xt, axis=1); el += np.linalg.norm(xl - xt, axis=1); cnt += 1
    return eb/cnt, el/cnt, cnt
print("\n[C] multi-step open-loop error in (sinθ,cosθ-1,θ̇):")
for reg in ["broad", "near-upright"]:
    eb, el, cnt = openloop(reg, 300, 40, 7)
    print(f"  [{reg:11s}] trajs={cnt}")
    for h in [5, 10, 20, 40]:
        print(f"     H={h:2d}  bilinear={eb[h]:.4f}  linear={el[h]:.4f}  ratio(lin/bi)={el[h]/max(eb[h],1e-9):.2f}x")
