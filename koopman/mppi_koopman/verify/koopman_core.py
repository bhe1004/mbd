"""Shared Koopman model algebra for RB-KMPPI verification (system-agnostic)."""
import numpy as np
from scipy.optimize import linprog

DT = 0.05

# ----------------------------------------------------------------- 2D Tier-0 systems
def f_ct_2d(x, u, sys):
    x1, x2 = x[0], x[1]
    if sys == "vanderpol":
        return np.array([x2, 1.0*(1.0 - x1*x1)*x2 - x1 + u])
    return np.array([x2, -np.sin(x1) - 0.2*x2 + np.cos(x1)*u])      # pendulum_sdg
def step_2d(x, u, sys):
    k1 = f_ct_2d(x, u, sys); k2 = f_ct_2d(x+0.5*DT*k1, u, sys)
    k3 = f_ct_2d(x+0.5*DT*k2, u, sys); k4 = f_ct_2d(x+DT*k3, u, sys)
    return x + (DT/6.0)*(k1+2*k2+2*k3+k4)
def Psi_2d(x):
    x1, x2 = x[0], x[1]
    return np.array([x1, x2, x1*x1, x1*x2, x2*x2, x1**3, x1*x1*x2, x1*x2*x2, x2**3])
N2 = 9
C2 = np.zeros((2, N2)); C2[0, 0] = 1.0; C2[1, 1] = 1.0

def gather_2d(sys, n_traj, L, u_amp, x_amp, seed):
    r = np.random.default_rng(seed); X, U, Y = [], [], []
    for _ in range(n_traj):
        x0 = r.uniform(-x_amp, x_amp, size=2); us = r.uniform(-u_amp, u_amp, size=L)
        xs = [x0.copy()]
        for u in us: xs.append(step_2d(xs[-1], u, sys))
        xs = np.array(xs)
        if np.max(np.abs(xs)) > 6.0: continue
        for k in range(L):
            X.append(Psi_2d(xs[k])); U.append([us[k]]); Y.append(Psi_2d(xs[k+1]))
    return np.array(X).T, np.array(U).T, np.array(Y).T          # (N,D),(m,D),(N,D)

# ------------------------------------------------------- generic bilinear/linear EDMDc
def identify(Zx, Uu, Zy):
    N, m = Zx.shape[0], Uu.shape[0]
    Th_lin = Zy @ np.linalg.pinv(np.vstack([Zx, Uu]))
    A_lin, B_lin = Th_lin[:, :N], Th_lin[:, N:]
    blocks = [Zx, Uu] + [Uu[i:i+1, :]*Zx for i in range(m)]
    Th = Zy @ np.linalg.pinv(np.vstack(blocks))
    A, B0 = Th[:, :N], Th[:, N:N+m]
    Bs = [Th[:, N+m+i*N: N+m+(i+1)*N] for i in range(m)]
    return dict(N=N, m=m, A=A, B0=B0, Bs=Bs, A_lin=A_lin, B_lin=B_lin)

def identify_ridge(Zx, Uu, Zy, gamma):
    """Ridge-regularized bilinear EDMDc — shrinks A toward 0, a principled stand-in for
    stability-constrained identification (SafEDMD). Larger gamma -> more Schur-stable A."""
    N, m = Zx.shape[0], Uu.shape[0]
    Phi = np.vstack([Zx, Uu] + [Uu[i:i+1, :]*Zx for i in range(m)])
    Th = (Zy @ Phi.T) @ np.linalg.inv(Phi @ Phi.T + gamma*np.eye(Phi.shape[0]))
    A, B0 = Th[:, :N], Th[:, N:N+m]
    Bs = [Th[:, N+m+i*N: N+m+(i+1)*N] for i in range(m)]
    Phil = np.vstack([Zx, Uu])
    Thl = (Zy @ Phil.T) @ np.linalg.inv(Phil @ Phil.T + gamma*np.eye(N+m))
    return dict(N=N, m=m, A=A, B0=B0, Bs=Bs, A_lin=Thl[:, :N], B_lin=Thl[:, N:])

def predict_bi(M, Zx, Uu):
    return M["A"]@Zx + M["B0"]@Uu + sum(Uu[i:i+1, :]*(M["Bs"][i]@Zx) for i in range(M["m"]))

def roll_bi(M, z0, Us):
    z = z0.copy(); Z = [z]
    for u in np.atleast_2d(Us):
        z = M["A"]@z + M["B0"]@u + sum(u[i]*(M["Bs"][i]@z) for i in range(M["m"])); Z.append(z)
    return np.array(Z)
def roll_lin(M, z0, Us):
    z = z0.copy(); Z = [z]
    for u in np.atleast_2d(Us):
        z = M["A_lin"]@z + M["B_lin"]@u; Z.append(z)
    return np.array(Z)

def fit_bound(M, Zx, Uu, Zy):
    R = Zy - predict_bi(M, Zx, Uu)
    rn, zn, un = np.linalg.norm(R, axis=0), np.linalg.norm(Zx, axis=0), np.linalg.norm(Uu, axis=0)
    res = linprog([1.0, 1.0], A_ub=-np.vstack([zn, un]).T, b_ub=-rn,
                  bounds=[(0, None), (0, None)], method="highs")
    return float(res.x[0]), float(res.x[1])

def mbar(M, u):                    # ||M(u)||_2 <= ||A|| + sum |u_i| ||B_i||   (eq 0.4)
    return np.linalg.norm(M["A"], 2) + sum(abs(u[i])*np.linalg.norm(M["Bs"][i], 2) for i in range(M["m"]))

def euclid_tube(M, zhat, Us, cx, cu):                          # eq 1.2
    e = 0.0; E = [0.0]
    for k, u in enumerate(np.atleast_2d(Us)):
        e = (mbar(M, u)+cx)*e + cx*np.linalg.norm(zhat[k]) + cu*np.linalg.norm(u); E.append(e)
    return np.array(E)
