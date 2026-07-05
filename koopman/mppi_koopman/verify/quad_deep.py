"""
3D QUADROTOR (MuJoCo) deep Koopman: deep-LINEAR (MPPI-DK) vs deep-BILINEAR (ours).
Thrust enters via attitude: world force = R @ [0,0,T] = T * (3rd column of R) -> STRONG state-input
coupling (like unicycle v*cosθ, but 3D). Linear Koopman (z+=Az+Bu) cannot represent T*R_up; bilinear
(z+=Az+B0u+Σ u_i B_i z) can. Expect a large multi-step prediction gap (the 3D analog of the unicycle 42–48×).
State/base features (18): pos(3), linvel(3), R_flat(9), angvel(3). Input u=(T,τx,τy,τz), m=4.
Output: out/quad_deep.png (+ printed multi-step pred err)
"""
import os, numpy as np, torch, torch.nn as nn, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

XML = """
<mujoco><option timestep="0.01" gravity="0 0 -9.81"/>
<worldbody><body name="quad" pos="0 0 1"><freejoint/>
<geom type="box" size="0.12 0.12 0.03" mass="1"/></body></worldbody></mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
bid = model.body("quad").id; MG = 9.81

def ctrl_step(u, nsub=2):
    for _ in range(nsub):
        R = data.xmat[bid].reshape(3, 3)
        data.xfrc_applied[bid, :3] = R @ np.array([0.0, 0.0, u[0]])
        data.xfrc_applied[bid, 3:] = R @ np.array([u[1], u[2], u[3]])
        mujoco.mj_step(model, data)

def feat():
    R = data.xmat[bid].reshape(3, 3)
    return np.concatenate([data.qpos[:3], data.qvel[:3], R.flatten(), data.qvel[3:6]])   # 18

def reset_rand(r):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = r.uniform([-2, -2, 0.5], [2, 2, 2.5])
    ax = r.normal(size=3); ax /= np.linalg.norm(ax)+1e-9; ang = r.uniform(-1.2, 1.2)   # AGGRESSIVE tilts
    data.qpos[3:7] = [np.cos(ang/2), *(np.sin(ang/2)*ax)]
    data.qvel[:3] = r.uniform(-2, 2, 3); data.qvel[3:6] = r.uniform(-2, 2, 3)
    mujoco.mj_forward(model, data)

def gen_snips(N, K, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, 18)); U = np.zeros((N, K, 4))
    for i in range(N):
        reset_rand(r); B[i, 0] = feat()
        for k in range(K):
            u = np.array([MG + r.uniform(-5, 5), *r.uniform(-1.0, 1.0, 3)]); U[i, k] = u   # aggressive ctrl
            ctrl_step(u); B[i, k+1] = feat()
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

K = 12
print("collecting MuJoCo quadrotor snippets..."); Btr, Utr = gen_snips(4000, K, 1); Bte, Ute = gen_snips(1000, K, 2)
ND = 18
class DK(nn.Module):
    def __init__(self, extra=6, hid=64, bilinear=False, m=4):
        super().__init__(); self.N = ND+extra; self.bilinear = bilinear; self.m = m
        self.enc = nn.Sequential(nn.Linear(ND, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+0.01*torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(0.01*torch.randn(self.N, m))
        if bilinear: self.B1 = nn.Parameter(0.01*torch.randn(m, self.N, self.N))
    def lift(self, b): return torch.cat([b, self.enc(b)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(self.m): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :ND]

def train(bilinear, epochs=160, bs=512, lr=1e-3):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), lr=lr); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.0
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + 0.1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    return m
print("training deep Koopman (linear & bilinear)..."); m_lin = train(False); m_bi = train(True)

@torch.no_grad()
def pred(m):
    z = m.lift(Bte[:, 0]); errs = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k]); errs.append(((m.decode(z)-Bte[:, k+1])**2).sum(-1).sqrt().mean().item())
    return errs
@torch.no_grad()
def pred_sl(m, sl):   # error on a feature slice (e.g. velocity 3:6 = where thrust coupling acts)
    z = m.lift(Bte[:, 0]); errs = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k]); errs.append(((m.decode(z)[:, sl]-Bte[:, k+1, sl])**2).sum(-1).sqrt().mean().item())
    return errs
el, eb = pred(m_lin), pred(m_bi)
vl, vb = pred_sl(m_lin, slice(3, 6)), pred_sl(m_bi, slice(3, 6))      # linear velocity
pl, pb = pred_sl(m_lin, slice(0, 3)), pred_sl(m_bi, slice(0, 3))      # position
print("[multi-step pred err]  full18 / velocity(3:6, coupling acts here) / position")
for h in [1, 4, 8, 12]:
    print(f"   H={h:2d}  full: lin={el[h-1]:.2f} bi={eb[h-1]:.2f} ({el[h-1]/max(eb[h-1],1e-9):.1f}x) | "
          f"vel: lin={vl[h-1]:.2f} bi={vb[h-1]:.2f} ({vl[h-1]/max(vb[h-1],1e-9):.1f}x) | "
          f"pos: lin={pl[h-1]:.2f} bi={pb[h-1]:.2f} ({pl[h-1]/max(pb[h-1],1e-9):.1f}x)")

plt.figure(figsize=(7, 4.6))
hs = range(1, K+1)
plt.plot(hs, el, "o-", label="linear (MPPI-DK) full"); plt.plot(hs, eb, "s-", label="bilinear (ours) full")
plt.plot(hs, vl, "o--", alpha=0.6, label="linear velocity"); plt.plot(hs, vb, "s--", alpha=0.6, label="bilinear velocity")
plt.xlabel("horizon"); plt.ylabel("open-loop prediction error"); plt.yscale("log")
plt.title("3D MuJoCo quadrotor: bilinear >> linear (thrust-via-attitude coupling)")
plt.legend(fontsize=8); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "quad_deep.png"), dpi=120)
print(f"outputs -> {OUT}/quad_deep.png")
