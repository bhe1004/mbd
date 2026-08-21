"""3D velocity-controlled drone: world position integrates body-frame velocity through yaw rotation,
p_world(k+1) = p + R(yaw)[vx,vy,vz] dt, yaw(k+1)=yaw+w dt. Input u=[vx,vy,vz,w] (m=4) enters the
tracked world position through an IRREDUCIBLE SO(3)/yaw rotation -> the 3D analogue of the unicycle.
Tests multi-step world-position prediction gap (linear vs bilinear deep Koopman)."""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
DT = 0.05; M = 4; ND = 6   # base obs = [x,y,z, sin yaw, cos yaw-1? -> use sinyaw, cosyaw], +... keep 6: x,y,z,sinψ,cosψ,zdummy? use 5
ND = 5                      # [x, y, z, sin(yaw), cos(yaw)]
def step(s, u):             # s=[x,y,z,yaw], u=[vx,vy,vz,w]
    x, y, z, ya = s; vx, vy, vz, w = u; c, sn = np.cos(ya), np.sin(ya)
    return np.array([x + (c*vx - sn*vy)*DT, y + (sn*vx + c*vy)*DT, z + vz*DT, ya + w*DT])
def obs(s): return np.array([s[0], s[1], s[2], np.sin(s[3]), np.cos(s[3])], np.float32)
K = 15
def gen(N, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, ND), np.float32); U = np.zeros((N, K, M), np.float32)
    for i in range(N):
        s = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-np.pi, np.pi)]); B[i, 0] = obs(s)
        bias = np.concatenate([r.uniform(-0.7, 0.7, 3), r.uniform(-1.2, 1.2, 1)])
        for k in range(K):
            u = bias + np.concatenate([r.uniform(-0.4, 0.4, 3), r.uniform(-0.6, 0.6, 1)]); U[i, k] = u
            s = step(s, u); B[i, k+1] = obs(s)
    return torch.tensor(B), torch.tensor(U)
print("generating drone data..."); Btr, Utr = gen(4000, 1); Bte, Ute = gen(1000, 2)
class DK(nn.Module):
    def __init__(self, extra=8, hid=64, bilinear=False):
        super().__init__(); self.N = ND+extra; self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(ND, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+.01*torch.randn(self.N, self.N)); self.B0 = nn.Parameter(.01*torch.randn(self.N, M))
        if bilinear: self.B1 = nn.Parameter(torch.zeros(M, self.N, self.N))
    def lift(self, b): return torch.cat([b, self.enc(b)], -1)
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
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + .1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    return m
print("training linear & bilinear..."); m_lin = train(False); m_bi = train(True)
@torch.no_grad()
def pos_err(m):
    z = m.lift(Bte[:, 0]); e = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k]); p = m.decode(z)[:, :3]
        e.append(((p-Bte[:, k+1, :3])**2).sum(-1).sqrt().mean().item())
    return e
el, eb = pos_err(m_lin), pos_err(m_bi)
print("\n[3D velocity-drone, multi-step WORLD-POSITION prediction error]:")
for h in [1, 5, 10, 15]:
    print(f"   H={h:2d}  linear={el[h-1]:.4f}  bilinear={eb[h-1]:.4f}  ratio={el[h-1]/max(eb[h-1],1e-9):.2f}x")
print(f"\nRESULT drone3d ratio@H15 = {el[-1]/max(eb[-1],1e-9):.2f}x  (>~5x = strong bilinear win like unicycle)")
