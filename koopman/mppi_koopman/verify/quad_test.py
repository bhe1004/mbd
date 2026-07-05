"""De-risk MuJoCo quadrotor: verify thrust-via-attitude control (hover + tilt-to-translate)."""
import numpy as np, mujoco

XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="quad" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.12 0.12 0.03" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
bid = model.body("quad").id
MG = 9.81  # mass*g

def ctrl_step(u, nsub=2):                       # u=(T, tx,ty,tz) thrust + body torques; control dt=0.02
    for _ in range(nsub):
        R = data.xmat[bid].reshape(3, 3)
        data.xfrc_applied[bid, :3] = R @ np.array([0.0, 0.0, u[0]])      # thrust along body-z
        data.xfrc_applied[bid, 3:] = R @ np.array([u[1], u[2], u[3]])    # body torque -> world
        mujoco.mj_step(model, data)

def reset(pos=(0, 0, 1)):
    mujoco.mj_resetData(model, data); data.qpos[:3] = pos; data.qpos[3] = 1.0  # identity quat
    mujoco.mj_forward(model, data)

# 1) hover: thrust = mg, no torque -> should stay put
reset()
for _ in range(100): ctrl_step([MG, 0, 0, 0])
print(f"[hover] pos={np.round(data.qpos[:3],3)}  (expect ~[0,0,1])  vel={np.round(data.qvel[:3],3)}")

# 2) tilt-to-translate: pitch torque briefly, then hover-thrust -> body tilts, thrust pushes sideways
reset()
for t in range(150):
    tau_y = 0.4 if t < 8 else (-0.4 if 8 <= t < 16 else 0.0)   # impulse then counter (settle a pitch)
    ctrl_step([MG, 0, tau_y, 0])
print(f"[tilt ] pos={np.round(data.qpos[:3],3)}  (expect x moved, z~kept)  "
      f"quat={np.round(data.qpos[3:7],3)}")

# 3) collect a random-control transition to confirm data pipeline
reset((0, 0, 2))
r = np.random.default_rng(0)
s_prev = np.concatenate([data.qpos.copy(), data.qvel.copy()])
ctrl_step([MG + r.uniform(-3, 3), *r.uniform(-0.3, 0.3, 3)])
s_now = np.concatenate([data.qpos.copy(), data.qvel.copy()])
print(f"[data ] qpos(7)+qvel(6) dim = {s_now.shape[0]};  moved by {np.round(s_now[:3]-s_prev[:3],4)}")
print("MuJoCo quadrotor OK")
