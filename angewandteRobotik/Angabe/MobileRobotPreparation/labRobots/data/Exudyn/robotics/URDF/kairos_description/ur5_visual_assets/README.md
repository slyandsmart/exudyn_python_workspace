# UR5 visual meshes

`ur5_visual_meshes.npz` contains the seven UR5 visual COLLADA meshes from
[Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description),
revision `0eb1a861f7772fceb38510d90fa2f98ec668070c`, converted to NumPy arrays.
Scene transforms, triangle indices, per-corner normals and diffuse material colors
are preserved. The adjacent LICENSE is the source repository's BSD-3-Clause license.

The visual offsets come from `config/ur5/visual_parameters.yaml` at the same
revision. `ur5_visual.py` maps the URDF frames to the existing simulation's
standard-DH frames; no URDF dynamics or collision geometry is used. Since 2026-11-05
the first DH offset is the URDF's 0.089159 m (previously 0.089459 m), so DH frame 6
equals the URDF tool0 used by MoveIt.

The visual meshes are enabled by default in `SimulationMobileRobot`. Pass
`useUR5Visuals=False` to restore the previous schematic arm rendering.
