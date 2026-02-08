from dm_control import suite
import cv2 as cv
import time
# Load CMU Humanoid environment
env = suite.load(domain_name="humanoid_CMU", task_name="run")  # CMU mocap included
phy = env.physics

pixels = phy.render(height=200, width=400, camera_id=0, render_context_offscreen=False)

while exit:
    # actions = data.NextActutatorValues()
    # if not actions == None:
        # actuator = env.CmuDataToCMUHumanoid(actions)
        # print(actuator)
        # env.step(actuator)
    
    # cv.imshow("Opencv",pixels)
    if cv.waitKey(1) & 0xFF == 27:  # backup ESC
        break

    time.sleep(0.01)
cv.destroyAllWindows()


print("Initial qpos:", phy.data.qpos)
print("Initial qvel:", phy.data.qvel)
