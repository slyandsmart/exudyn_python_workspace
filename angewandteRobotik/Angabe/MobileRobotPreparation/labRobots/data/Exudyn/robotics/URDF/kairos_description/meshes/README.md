# Kairos meshes of the MoveIt/URDF robot description

STL meshes of the robot description used by MoveIt on the real Kairos
(`summit_xls_171220A.urdf.xacro`, repository `git.uibk.ac.at/mekt/mobroblecture`,
commit `d2fe784adec01af9588014e2d96ca3f719e73784`). They are copied unchanged; the
poses at which they are mounted are in `labRobots/kairos/geometry.py` (`kairosParts`,
gripper constants), taken from the URDF frames relative to `summit_xl_base_footprint`.

| file | source package | URDF link | unit |
| --- | --- | --- | --- |
| `summit_xls_chassis.stl` | summit_xl_description/meshes/bases/hls | summit_xl_base_link | m |
| `structure_rear_velodyne.stl` | summit_xl_description/meshes/structures | summit_xl_rear_velodyne_structure_link | m |
| `velodyne_vlp16.stl` | robotnik_sensors/meshes | summit_xl_velodyne_base_link | m |
| `sick_s300.stl` | robotnik_sensors/meshes | summit_xl_front/rear_laser_base_link | m |
| `orbbec_astra.stl` | robotnik_sensors/meshes | summit_xl_front_rgbd_camera_link | m |
| `WSG50_110.stl` | wsg_50_simulation/meshes | summit_xl_wsg50_base_link | m |
| `GUIDE_WSG50_110.stl` | wsg_50_simulation/meshes | summit_xl_wsg50_gripper_left/right | mm |
| `WSG-FMF.stl` | wsg_50_simulation/meshes | summit_xl_wsg50_finger_left/right | mm |

All three packages are BSD licensed (`package.xml`); the Robotnik license text is in
`LICENSE_robotnik`. The omni wheel meshes of the URDF are COLLADA only; the simulation
keeps its own mecanum wheel graphics, whose size (radius 0.127 m, width 0.1 m) equals
the URDF visual wheel mesh.
