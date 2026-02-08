from dm_control.locomotion.mocap.loader import HDF5TrajectoryLoader
from dm_control.locomotion.mocap.cmu_mocap_data import *
import numpy as np
dataloader = HDF5TrajectoryLoader(path="C:/Users/nadhu/miniconda3/envs/robo/lib/site-packages/dm_control/locomotion/mocap/cmu_2020_dfe3e9e0.h5")
keys  = dataloader.keys()
print(keys)
traj = dataloader.get_trajectory(key="CMU_001_01").as_dict()
#walker
if "walker/position" in traj:
    print("position")
    arr = np.array(traj["walker/position"])
    print(arr.shape)
if "walker/quaternion" in traj:
    print("quaternion")
    arr = np.array(traj["walker/quaternion"])
    print(arr.shape)
if "walker/joints" in traj:
    print("joints")
    arr = np.array(traj["walker/joints"])
    print(arr.shape)

if "walker/end_effectors" in traj:
    print("end_effectors")
    arr = np.array(traj["walker/end_effectors"])
    print(arr.shape)

if "walker/angular_velocity" in traj:
    print("angular_velocity")
    arr = np.array(traj["walker/angular_velocity"])
    print(arr.shape)

if "walker/joints_velocity" in traj:
    print("joints_velocity")
    arr = np.array(traj["walker/joints_velocity"])
    print(arr.shape)

if "walker/appendages" in traj:
    print("appendages")
    arr = np.array(traj["walker/appendages"])
    print(arr.shape)

if "walker/body_positions" in traj:
    print("body_positions")
    arr = np.array(traj["walker/body_positions"])
    print(arr.shape)

if "walker/body_quaternions" in traj:
    print("body_quaternions")
    arr = np.array(traj["walker/body_quaternions"])
    print(arr.shape)


# print(traj)