"""Rollout wall-time: bilinear Koopman (matvec) vs true-dynamics integration vs a physics engine.
Justifies using a learned Koopman model over the true-dynamics oracle. Quadrotor-scale settings."""
import numpy as np, torch, time, mujoco
torch.manual_seed(0); torch.set_num_threads(4)
Ks, T, m, N, NSUB = 1024, 30, 4, 24, 2
G, DT, MASS = 9.81, 0.02, 1.0
INER = torch.tensor([0.0051, 0.0051, 0.0096])

# --- bilinear Koopman rollout (batched matvec) ---
A = torch.randn(N, N)*0.1 + torch.eye(N); B0 = torch.randn(N, m)*0.1; B1 = torch.randn(m, N, N)*0.05
@torch.no_grad()
def koopman(U):
    z = torch.randn(Ks, N)
    for t in range(T):
        u = U[:, t]; z = z @ A.T + u @ B0.T
        for i in range(m): z = z + u[:, i:i+1]*(z @ B1[i].T)
    return z

# --- analytic true dynamics (batched: Rodrigues SO(3) + Newton-Euler) ---
def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1), torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)
def exp_so3(w):
    a = w.norm(dim=-1, keepdim=True); K = skew(w/(a+1e-9)); aa = a.unsqueeze(-1)
    return torch.eye(3).expand(K.shape) + torch.sin(aa)*K + (1-torch.cos(aa))*(K @ K)
@torch.no_grad()
def analytic(U):
    P = torch.randn(Ks, 3); V = torch.randn(Ks, 3); R = torch.eye(3).repeat(Ks, 1, 1); W = torch.randn(Ks, 3)*0.1
    for t in range(T):
        u = U[:, t]
        for _ in range(NSUB):
            acc = R[:, :, 2]*u[:, 0:1]/MASS + torch.tensor([0., 0., -G])
            wdot = (u[:, 1:4] - torch.cross(W, INER*W, dim=-1))/INER
            W = W + wdot*DT; V = V + acc*DT; R = R @ exp_so3(W*DT); P = P + V*DT
    return P

# --- physics engine (MuJoCo), per-sample sequential (cannot batch on CPU) ---
XML = """<mujoco><option timestep="0.01" gravity="0 0 -9.81"/><worldbody>
<body name="q" pos="0 0 1"><freejoint/><geom type="box" size="0.12 0.12 0.03" mass="1"/></body></worldbody></mujoco>"""
mj = mujoco.MjModel.from_xml_string(XML); md = mujoco.MjData(mj)
def mujoco_seq(nsamp=Ks):
    for _ in range(nsamp):
        mujoco.mj_resetData(mj, md)
        for _ in range(T*NSUB): mujoco.mj_step(mj, md)

U = torch.randn(Ks, T, m)*0.3
koopman(U); analytic(U)                                  # warmup
def bench(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps): fn(U)
    return (time.perf_counter()-t0)/reps*1e3              # ms per control step
tk = bench(koopman, 30); ta = bench(analytic, 30)
t0 = time.perf_counter(); mujoco_seq(); tm = (time.perf_counter()-t0)*1e3
print(f"rollout cost per control step  (Ks={Ks}, T={T}):")
print(f"  bilinear Koopman (batched matvec) : {tk:7.2f} ms")
print(f"  true dynamics (batched analytic)  : {ta:7.2f} ms   ({ta/tk:.1f}x Koopman)")
print(f"  physics engine (MuJoCo, per-sample): {tm:7.1f} ms   ({tm/tk:.0f}x Koopman)")
