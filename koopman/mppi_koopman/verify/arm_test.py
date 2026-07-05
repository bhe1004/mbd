"""De-risk: self-contained 7-DOF arm in MuJoCo, verify forward kinematics (ee site moves with q)."""
import numpy as np, mujoco

XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="b1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.04"/>
      <body name="b2" pos="0 0 0.18">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.038"/>
        <body name="b3" pos="0 0 0.18">
          <joint name="j3" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.035"/>
          <body name="b4" pos="0 0 0.16">
            <joint name="j4" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.033"/>
            <body name="b5" pos="0 0 0.16">
              <joint name="j5" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.03"/>
              <body name="b6" pos="0 0 0.14">
                <joint name="j6" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.028"/>
                <body name="b7" pos="0 0 0.12">
                  <joint name="j7" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.025"/>
                  <site name="ee" pos="0 0 0.12" size="0.03" rgba="1 0 0 1"/>
                </body></body></body></body></body></body></body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
eid = model.site("ee").id
print(f"nq={model.nq} nv={model.nv} (expect 7)")

def fk(q):
    data.qpos[:] = q; mujoco.mj_forward(model, data); return data.site_xpos[eid].copy()

print(f"[home q=0] ee = {np.round(fk(np.zeros(7)),3)}  (fully extended up, z~ sum of links)")
r = np.random.default_rng(0)
for _ in range(3):
    q = r.uniform(-1, 1, 7); print(f"[rand] q={np.round(q,2)} -> ee={np.round(fk(q),3)}")
# reach span check
ees = np.array([fk(r.uniform(-2, 2, 7)) for _ in range(2000)])
print(f"[span] ee x:[{ees[:,0].min():.2f},{ees[:,0].max():.2f}] y:[{ees[:,1].min():.2f},{ees[:,1].max():.2f}] z:[{ees[:,2].min():.2f},{ees[:,2].max():.2f}]")
print("arm FK OK")
