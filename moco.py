import torch
import gymnasium as gym
import torch.nn as nn
from env import HumanoidEnvironment
import torchvision.models as models
from env import CMUTrajectory , HumanoidEnvironment
from torchvision import transforms
vgg16 = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
import cv2 as cv
# cv.namedWindow("training")
# cv.namedWindow("actual")

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class MotionController(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(vgg16.features,vgg16.avgpool,nn.Flatten())
        self.obs = nn.Sequential(
                      nn.Linear(7*7*512 + 63,1024),
                      nn.Tanh(),
                      nn.Linear(1024,512),
                      nn.Tanh(),
                      nn.Linear(512,112),
                      nn.Tanh()
                    )
        self.decoder = nn.Sequential(
                        nn.Linear(112,512),
                        nn.Tanh(),
                        nn.Linear(512,256),
                        nn.Tanh(),
                        nn.Linear(256,63),
                        nn.Tanh()
                    )
        self.optimizer = torch.optim.Adam(self.parameters(),lr=2e-5,eps=3e-9)
        self.lossFunction = nn.MSELoss()

    def forward(self,image,x_obs):
        if(x_obs.shape[1] != 63):
            return None
        feat = self.feature(image)
        inp = torch.cat((feat,x_obs),dim=1)
        out = self.obs(inp)
        mean = out[..., :56]
        std  = torch.nn.functional.softplus(out[..., 56:112]) + 1e-30
        
        eps = torch.randn_like(mean)
        actions = mean + std * eps
        return (self.decoder(out),actions)
    
    def Loss(self,actual_output,pred_out):
        return 
    
class Train:
    def __init__(self):
        self.model = MotionController()
        self.data = CMUTrajectory()
        self.env = HumanoidEnvironment()
        self.env.ActutatorNames()
        self.lowact = torch.tensor(self.env.low_act).to(device=device)
        self.highact = torch.tensor(self.env.high_act).to(device=device)
        self.model.to(device=device)
        self.env.reset()

    def Start(self,num_iter=100):
        i=-1
        while num_iter>0:
            init_state = self.env.physics.data.qpos
            init_state = torch.tensor(data=init_state,dtype=torch.float32)
            init_image = self.env.physics.render(height=224,width=224,camera_id="walker/egocentric")

            img = torch.tensor(init_image.copy(),dtype=torch.float32)
            img = img.permute(2,0,1)
            img = transform(img/255.0)
            img = img.unsqueeze(dim=0)
            init_state = init_state.unsqueeze(dim=0)
            i+=1
            (pred_state,actions) = self.model(img.to(device=device),init_state.to(device=device))
            # print(actions.shape)
            # print("actions nan :",actions.isnan().any())
            # print("actions inf :",actions.isinf().any())
            self.env.physics.data.ctrl[:] = torch.clip(actions.detach(),self.lowact,self.highact).cpu().numpy()[0][:]
            try:
                self.env.physics.step()
            except Exception:
                i=-1
                num_iter-=1
                self.env.physics.reset()
                continue
                
            # cv.imshow("training",self.env.physics.render(height=400,width=500,camera_id="follow"))
            self.data.LoadPoint(self.env.physics,i)
            actual_qpos = self.env.physics.data.qpos
            actual_qpos = torch.tensor(actual_qpos,dtype=torch.float32).reshape(shape=(1,63))
            # print("preshape",pred_state.shape)
            # print("actual shape ",actual_qpos.shape)
            loss = self.model.lossFunction(pred_state,actual_qpos.to(device=device))
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            print("loss:",loss)
            # print("num_points : i ",self.data.numpoints)
            # print("i : ",i)
            if i == self.data.numpoints:
                i=-1
                self.env.reset()
                num_iter-=1
            # key = cv.waitKey(1)
            # if key == ord("z"):
            #     break

        # cv.destroyAllWindows()
        # img = self.env.physics.render(height=224,width=224,camera_id="walker/egocentric")
        # # img = torch.from_numpy(img.copy())
        # img = torch.tensor(img.copy(),dtype=torch.float32)
        # img = img.permute(2,0,1)
        # img = transform(img)
        # img = img.unsqueeze(dim=0)
        
        
        # obs = torch.tensor(self.env.physics.data.qpos[:],dtype=torch.float32).unsqueeze(dim=0)
        # output = self.model(img,obs)



if __name__ == "__main__":
    train = Train()
    train.Start()



        




        