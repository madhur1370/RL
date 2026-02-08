from dm_control import mjcf, viewer
from dm_control import mujoco as mj
import numpy as np
import cv2 as cv
import keyboard as kb
import time
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.walkers import CMUHumanoidPositionControlledV2020
from dm_control.locomotion.mocap.loader import HDF5TrajectoryLoader
from dm_control.locomotion.mocap.cmu_mocap_data import *
from dm_control.suite.humanoid_CMU import HumanoidCMU

qpos = np.zeros(shape=(63),dtype=np.float64)
qpos[0:7] = [0 ,0 ,1, 0.859, 1.   , 1.   , 0.859]

class HumanoidEnvironment:
    def __init__(self):
        self.arena  = floors.Floor()
        self.human = CMUHumanoidPositionControlledV2020()
        self.arena.add_free_entity(self.human)
        self.arena.mjcf_model.worldbody.add(
                    "camera",
                    name="follow",
                    mode="track",
                    target="walker/root",
                    pos=[0, 0, 5],   # behind & above humanoid
                    fovy=90
                    )
        self.physics = mjcf.Physics.from_mjcf_model(self.arena.mjcf_model)

    def NumActuator(self):
        return self.physics.model.nu
    
    def reset(self):
        self.physics.reset()
        self.physics.data.qpos[:] = qpos
        self.physics.data.qvel[:] = 0

    def step(self,arr:np.array): #56 actuators or motors
        arr = np.random.uniform(-1.0, 1.0, size=56)
        self.physics.set_control(arr)
        self.physics.step()
    

    def ActutatorNames(self):
        for i in range(self.physics.model.nu):
            name = self.physics.model.id2name(i, mj.mjtObj.mjOBJ_ACTUATOR)
            ctrl_low, ctrl_high = self.physics.model.actuator_ctrlrange[i]
            print(i, name, ctrl_low, ctrl_high)
        print("\n")

    def WalkerPos(self):
        model = self.physics.model
        for jid in range(model.njnt):
            name = model.id2name(jid, mj.mjtObj.mjOBJ_JOINT)
            qadr = model.jnt_qposadr[jid]
            jtype = model.jnt_type[jid]
        
            if jtype == mj.mjtJoint.mjJNT_FREE:
                size = 7
            elif jtype == mj.mjtJoint.mjJNT_BALL:
                size = 4
            else:
                size = 1
        
            print(f"{name:15s}  qpos[{qadr}:{qadr+size}]  type={jtype} range = {model.jnt_range[jid] }")
        
    def SetQpos(self):
        self.physics.data.qpos[:]=[0 ,0 ,1 ]
        return 

    def CmuDataToCMUHumanoid(self,arr):
        if "" in arr:
            return None
        actions = np.zeros(shape=(56),dtype=np.float64)
        actions[0:3] = arr["head"][:] / 40.0
        actions[3:5] = arr["lclavicle"][:] /80.0
        actions[5:8] = arr["lfemur"][:] 
        actions[5] /= 300.0
        actions[6:8] /= 200.0
        actions[8:9] = arr["lfingers"][:] / 20.0
        actions[9:11] = arr["lfoot"][:]  
        actions[9] /= 120.0
        actions[10] /= 50.0
        actions[11:13] = arr["lhand"][:] / 20.0
        actions[13:16] = arr["lhumerus"][:] / 120.0

        actions[16:19] = arr["lowerback"][:]
        actions[16] /= 300.0
        actions[17] /= 180.0
        actions[18] /= 200.0
        actions[19:22] = arr["lowerneck"][:] / 120.0  
        
        actions[22:23] = arr["lradius"][:] / 90
        actions[23:25] = arr["lthumb"][:] / 20.0
        actions[25:26] = arr["ltibia"][:] /160
        actions[26:27] = arr["ltoes"][:]   / 20.0
        actions[27:28] = arr["lwrist"][:] / 20.0
        
        actions[28:30] = arr["rclavicle"][:] /80.0
        actions[30:33] = arr["rfemur"][:]
        actions[30] /= 300.0
        actions[31:33] /= 200.0

        actions[33:34] = arr["rfingers"][:] / 20.0
        actions[34:36] = arr["rfoot"][:]  
        actions[34] /= 120.0
        actions[35] /= 50.0
        actions[36:38] = arr["rhand"][:] / 20.0
        actions[38:41] = arr["rhumerus"][:]  / 120.0
        actions[41:42] = arr["rradius"][:] / 90
        actions[42:44] = arr["rthumb"][:] / 20.0
        actions[44:45] = arr["rtibia"][:] /160
        actions[45:46] = arr["rtoes"][:]  / 20.0
        actions[46:47] = arr["rwrist"][:] / 20.0

        actions[47:50] = arr["thorax"][:] 
        actions[47] /= 300.0
        actions[48] /= 80.0
        actions[49] /= 200.0
        actions[50:53] = arr["upperback"][:]
        actions[50] /= 300.0
        actions[51] /= 180.0
        actions[52] /= 200.0
        actions[53:56] = arr["upperneck"][:] /60.0

        return actions


class CmuDataset:
    """ dataset file should be like this only
        1
        root 7.48825 15.9816 -35.4705 6.80919 6.43628 3.66541
        lowerback -5.29073 -3.44697 -4.4228
        upperback -1.26812 -4.39824 2.15759
        thorax 2.1545 -2.22836 4.38208
        lowerneck -22.1525 -1.97402 -11.5654
        upperneck 18.8967 -0.970822 8.78318
        head 9.73703 -0.263706 4.88753
        rclavicle 5.90887e-014 -1.03368e-014
        rhumerus -48.0525 12.9166 -87.4019
        rradius 28.8482
        rwrist -11.0895
        rhand -25.3172 -27.317
        rfingers 7.12502
        rthumb 1.2052 -57.2933
        lclavicle 5.90887e-014 -1.03368e-014
        lhumerus -9.75298 -26.8186 87.5251
        lradius 35.7184
        lwrist 8.15954
        lhand -15.1223 -25.8408
        lfingers 7.12502
        lthumb 11.0468 3.84855
        rfemur -39.6009 8.70813 9.93913
        rtibia 26.9485
        rfoot -21.3669 -11.8756
        rtoes -14.0976
        lfemur 9.73125 4.9248 -28.1781
        ltibia -3.18055e-015
        lfoot -12.2814 -3.36507
        ltoes -5.33983
        2
    """
    def __init__(self,filename:str,chunk:int,numactuators:int):
        self.filename = filename
        self.f = open(self.filename,mode="r")
        self.one_chunk_size = chunk
        self.num_data = 0
        self.num_actuators = numactuators
        
    def ReadData(self):
        arr = {}
        t=0
        for i in range(0,self.one_chunk_size,1):
            line = self.f.readline().rstrip("\n").split(" ")
            if i<=1:
                continue
            if line == "":
                self.f.close()
                return None
            linesize = len(line)
            temp = np.zeros(shape=(linesize -1),dtype=np.float64)

            for j in range(1,linesize):
                temp[j-1] = np.float64(line[j])
            arr[line[0]] = temp
        self.num_data+=1
        return arr
    
    def ReRead(self):
        self.f = open(self.filename,"r")
    
    def NextActutatorValues(self):
        return self.ReadData()


class CMUTrajectory : 
    def __init__(self):
        self.dataloader = HDF5TrajectoryLoader(path="C:/Users/nadhu/miniconda3/envs/robo/lib/site-packages/dm_control/locomotion/mocap/cmu_2020_dfe3e9e0.h5")
        self.keys  = self.dataloader.keys()
        # jumping 1 walking 5
        self.traj = self.dataloader.get_trajectory(key="CMU_005_01").as_dict()
        self.numpoints = np.array(self.traj["walker/position"]).shape[0]
    
    def LoadPoint(self,physics,i):
        physics.reset()
        physics.data.qpos[7:] = self.traj["walker/joints"][i]
        physics.data.qpos[0:3] = self.traj["walker/position"][i][:]
        physics.data.qpos[3:7] = self.traj["walker/quaternion"][i][:]
        physics.forward()



exit = True
def stop_loop():
    global exit
    exit = False



env = HumanoidEnvironment()

env.reset()
phy = env.physics
trj = CMUTrajectory()
kb.add_hotkey("esc",stop_loop)

#make window for opencv
cv.namedWindow("Opencv",cv.WINDOW_NORMAL)
cv.resizeWindow("Opencv",800,400)
step_counter = 0
i = 0
while exit:
    trj.LoadPoint(physics=phy,i=i)
    i+=1
    if i==trj.numpoints: i=0
    # render every 2-3 steps instead of every single step
    if step_counter % 2 == 0:
        cam_arr = phy.data.qpos[0:3] + np.array([0,0,1])
        pixels = phy.render(height=200, width=400,camera_id="follow")
        cv.imshow("Opencv", pixels)

    if cv.waitKey(1) & 0xFF == 27:
        break

    step_counter += 1
    time.sleep(0.005)  # smaller sleep
