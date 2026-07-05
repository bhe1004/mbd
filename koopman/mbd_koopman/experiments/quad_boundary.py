"""Boundary case: 3D MuJoCo quadrotor, where the coupling is second order.

The thrust command enters through the attitude (world force = T * R e_3), so
the state-input coupling acts on the *acceleration*, not directly on the
tracked output increment -- the complementary regime to the first-order
coupled systems (unicycle heading, manipulator Jacobian) that motivate the
bilinear rollout of BK-MBD. Following the same protocol as the manipulator
prediction study, deep-linear and deep-bilinear Koopman models are trained
identically and compared on open-loop prediction error vs. horizon, on the
full 18-dim feature vector and on the linear-velocity rows (the rows a
position controller consumes).

Expected outcome (delimits the BK-MBD regime): bilinear is clearly better on
the full state (it captures thrust-attitude coupling), but on the velocity
rows the two classes coincide -- a linear lift suffices there.

Output: paper/figs/quad_boundary.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.labelsize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

XML = """
<mujoco><option timestep="0.01" gravity="0 0 -9.81"/>
<worldbody><body name="quad" pos="0 0 1"><freejoint/>
<geom type="box" size="0.12 0.12 0.03" mass="1"/></body></worldbody></mujoco>
"""
MG = 9.81
ND = 18
K_TRAIN = 12
SEEDS = [0, 1, 2]

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
bid = model.body("quad").id


def ctrl_step(u, nsub=2):
    for _ in range(nsub):
        rot = data.xmat[bid].reshape(3, 3)
        data.xfrc_applied[bid, :3] = rot @ np.array([0.0, 0.0, u[0]])
        data.xfrc_applied[bid, 3:] = rot @ np.array([u[1], u[2], u[3]])
        mujoco.mj_step(model, data)


def feat():
    rot = data.xmat[bid].reshape(3, 3)
    return np.concatenate([data.qpos[:3], data.qvel[:3], rot.flatten(), data.qvel[3:6]])


def reset_rand(r):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = r.uniform([-2, -2, 0.5], [2, 2, 2.5])
    ax = r.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-9
    ang = r.uniform(-1.2, 1.2)
    data.qpos[3:7] = [np.cos(ang / 2), *(np.sin(ang / 2) * ax)]
    data.qvel[:3] = r.uniform(-2, 2, 3)
    data.qvel[3:6] = r.uniform(-2, 2, 3)
    mujoco.mj_forward(model, data)


def gen_snips(n, k, seed):
    r = np.random.default_rng(seed)
    b = np.zeros((n, k + 1, ND))
    u = np.zeros((n, k, 4))
    for i in range(n):
        reset_rand(r)
        b[i, 0] = feat()
        for j in range(k):
            uj = np.array([MG + r.uniform(-5, 5), *r.uniform(-1.0, 1.0, 3)])
            u[i, j] = uj
            ctrl_step(uj)
            b[i, j + 1] = feat()
    return (
        torch.tensor(b, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32),
    )


class QuadDK(nn.Module):
    def __init__(self, extra=6, hid=64, bilinear=False, m=4):
        super().__init__()
        self.N = ND + extra
        self.bilinear = bilinear
        self.m = m
        self.enc = nn.Sequential(
            nn.Linear(ND, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, extra),
        )
        self.A = nn.Parameter(torch.eye(self.N) + 0.01 * torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(0.01 * torch.randn(self.N, m))
        if bilinear:
            self.B1 = nn.Parameter(torch.zeros(m, self.N, self.N))

    def lift(self, b):
        return torch.cat([b, self.enc(b)], -1)

    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(self.m):
                zn = zn + u[..., i : i + 1] * (z @ self.B1[i].T)
        return zn

    def decode(self, z):
        return z[..., :ND]


def train(bilinear, seed, btr, utr, device, epochs=160, bs=512):
    torch.manual_seed(seed)
    m = QuadDK(bilinear=bilinear).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = btr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            b, u = btr[idx], utr[idx]
            z = m.lift(b[:, 0])
            loss = 0.0
            for k in range(K_TRAIN):
                z = m.fstep(z, u[:, k])
                loss = loss + ((m.decode(z) - b[:, k + 1]) ** 2).mean() \
                    + 0.1 * ((z - m.lift(b[:, k + 1]).detach()) ** 2).mean()
            opt.zero_grad()
            (loss / K_TRAIN).backward()
            opt.step()
    return m


@torch.no_grad()
def pred_slice(m, bte, ute, sl):
    z = m.lift(bte[:, 0])
    errs = []
    for k in range(K_TRAIN):
        z = m.fstep(z, ute[:, k])
        errs.append(
            ((m.decode(z)[:, sl] - bte[:, k + 1, sl]) ** 2)
            .sum(-1).sqrt().mean().item()
        )
    return np.array(errs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "figs"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("collecting MuJoCo quadrotor snippets ...", flush=True)
    btr, utr = gen_snips(4000, K_TRAIN, 1)
    bte, ute = gen_snips(1000, K_TRAIN, 2)
    btr, utr = btr.to(device), utr.to(device)
    bte, ute = bte.to(device), ute.to(device)

    curves = {k: [] for k in ("lin_full", "bil_full", "lin_vel", "bil_vel")}
    for seed in SEEDS:
        m_lin = train(False, seed, btr, utr, device)
        m_bil = train(True, seed, btr, utr, device)
        curves["lin_full"].append(pred_slice(m_lin, bte, ute, slice(0, ND)))
        curves["bil_full"].append(pred_slice(m_bil, bte, ute, slice(0, ND)))
        curves["lin_vel"].append(pred_slice(m_lin, bte, ute, slice(3, 6)))
        curves["bil_vel"].append(pred_slice(m_bil, bte, ute, slice(3, 6)))
        lf, bf = curves["lin_full"][-1], curves["bil_full"][-1]
        lv, bv = curves["lin_vel"][-1], curves["bil_vel"][-1]
        print(
            f"seed={seed}  H12 full: lin={lf[-1]:.2f} bil={bf[-1]:.2f} "
            f"({lf[-1]/bf[-1]:.1f}x) | vel: lin={lv[-1]:.2f} bil={bv[-1]:.2f} "
            f"({lv[-1]/bv[-1]:.1f}x)",
            flush=True,
        )

    hs = np.arange(1, K_TRAIN + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, color, marker, ls, alpha, label in (
        ("lin_full", "#1f77b4", "o", "-", 1.0, "linear (DK) --- full state"),
        ("bil_full", "#d62728", "s", "-", 1.0, "bilinear (BK) --- full state"),
        ("lin_vel", "#1f77b4", "o", "--", 0.5, "linear (DK) --- velocity"),
        ("bil_vel", "#d62728", "s", "--", 0.5, "bilinear (BK) --- velocity"),
    ):
        arr = np.array(curves[key])
        ax.plot(
            hs, arr.mean(0), marker=marker, ms=4, lw=2 if ls == "-" else 1.4,
            ls=ls, alpha=alpha, color=color, label=label,
        )
        ax.fill_between(
            hs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
            color=color, alpha=0.12, lw=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("horizon")
    ax.set_ylabel("open-loop prediction error")
    ax.legend(loc="center right")
    fig.tight_layout()
    path = args.output_dir / "quad_boundary.png"
    fig.savefig(path, dpi=200)
    print(f"saved={path}")


if __name__ == "__main__":
    main()
