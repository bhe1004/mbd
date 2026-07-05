"""
7-DOF arm (MuJoCo) deep Koopman, KINEMATIC velocity control (input u=q̇, m=7):
  q+ = q + q̇·dt  (trivially linear) ;  ee+ = FK(q+) ≈ ee + J(q)·q̇·dt  (BILINEAR: q̇_i × J_i(q)).
End-effector motion has 1st-order configuration-dependent coupling via the Jacobian J(q) — the
manipulator analog of the unicycle v·cosθ. Linear Koopman (constant B) cannot represent J(q)·q̇; bilinear can.
Expect a LARGE end-effector prediction gap (our regime, unlike the 2nd-order quadrotor).
Output: out/arm_deep.png (+ printed multi-step pred err: ee vs joints)
"""
import os, numpy as np, torch, torch.nn as nn, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
XML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm.xml")).read() if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm.xml")) else None
if XML is None:
    XML = """<mujoco><option timestep="0.01" gravity="0 0 0"/><worldbody>
    <body name="b1" pos="0 0 0.1"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.04"/>
    <body name="b2" pos="0 0 0.18"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.038"/>
    <body name="b3" pos="0 0 0.18"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.035"/>
    <body name="b4" pos="0 0 0.16"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.033"/>
    <body name="b5" pos="0 0 0.16"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.03"/>
    <body name="b6" pos="0 0 0.14"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.028"/>
    <body name="b7" pos="0 0 0.12"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.025"/>
    <site name="ee" pos="0 0 0.12" size="0.03"/></body></body></body></body></body></body></body></worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model); eid = model.site("ee").id
def fk(q):
    data.qpos[:] = q; mujoco.mj_forward(model, data); return data.site_xpos[eid].copy()

DT, QV, QLIM = 0.05, 2.0, 2.6
K = 15
def gen(N, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, 10)); U = np.zeros((N, K, 7))
    for i in range(N):
        q = r.uniform(-2, 2, 7); qd_c = r.uniform(-QV*0.6, QV*0.6, 7)          # coherent velocity bias
        B[i, 0] = np.concatenate([q, fk(q)])
        for k in range(K):
            qd = np.clip(qd_c + r.uniform(-QV*0.5, QV*0.5, 7), -QV, QV); U[i, k] = qd
            q = np.clip(q + qd*DT, -QLIM, QLIM); B[i, k+1] = np.concatenate([q, fk(q)])
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
print("collecting arm snippets..."); Btr, Utr = gen(4000, 1); Bte, Ute = gen(1000, 2)

ND, M = 10, 7
class DK(nn.Module):
    def __init__(self, extra=10, hid=96, bilinear=False):
        super().__init__(); self.N = ND+extra; self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(14, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+0.01*torch.randn(self.N, self.N)); self.B0 = nn.Parameter(0.01*torch.randn(self.N, M))
        if bilinear: self.B1 = nn.Parameter(torch.zeros(M, self.N, self.N))
    def lift(self, b):
        q = b[..., :7]; inp = torch.cat([torch.sin(q), torch.cos(q)], -1)
        return torch.cat([b, self.enc(inp)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(M): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :ND]
def train(bilinear, epochs=200, bs=512):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
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
def pred_sl(m, sl):
    z = m.lift(Bte[:, 0]); errs = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k]); errs.append(((m.decode(z)[:, sl]-Bte[:, k+1, sl])**2).sum(-1).sqrt().mean().item())
    return errs
el_ee, eb_ee = pred_sl(m_lin, slice(7, 10)), pred_sl(m_bi, slice(7, 10))      # end-effector (J(q) coupling)
el_q, eb_q = pred_sl(m_lin, slice(0, 7)), pred_sl(m_bi, slice(0, 7))          # joints (trivially linear)
print("[multi-step pred err]  end-effector (J coupling) / joints (trivial)")
for h in [1, 4, 8, 12]:
    print(f"   H={h:2d}  EE: lin={el_ee[h-1]:.4f} bi={eb_ee[h-1]:.4f} ({el_ee[h-1]/max(eb_ee[h-1],1e-9):.1f}x) | "
          f"joints: lin={el_q[h-1]:.4f} bi={eb_q[h-1]:.4f}")

plt.figure(figsize=(7, 4.6)); hs = range(1, K+1)
plt.plot(hs, el_ee, "o-", lw=2, label="linear (MPPI-DK) — end-effector"); plt.plot(hs, eb_ee, "s-", lw=2, label="bilinear (ours) — end-effector")
plt.plot(hs, el_q, "o--", alpha=0.5, label="linear — joints"); plt.plot(hs, eb_q, "s--", alpha=0.5, label="bilinear — joints")
plt.xlabel("horizon"); plt.ylabel("open-loop prediction error"); plt.yscale("log")
plt.title("7-DOF arm: bilinear >> linear on end-effector (Jacobian J(q) coupling, 1st-order)")
plt.legend(fontsize=8); plt.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "arm_deep.png"), dpi=120)
print(f"outputs -> {OUT}/arm_deep.png")
