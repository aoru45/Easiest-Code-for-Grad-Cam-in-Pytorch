import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
import cv2 as cv
import numpy as np
# 今日任务: GradCam
# 今日任务: Opencv推理caffe
class Hook():
    # 先建立Hook反向传播钩子和得到前向的featuremap
    def __init__(self,model):
        self.model = model
        self.feature = None
        self.grad = None
    def hook_backward(self,grad):
        self.grad = grad
    def __call__(self,x):
        for name, module in self.model.named_children():
            if name == "fc":
                x = x.view((-1,512))
                x = module(x)
            else:
                x = module(x)
            if name == "layer4":
                
                x.register_hook(self.hook_backward)
                self.feature = x
        return x
def show(img, mask):
    img = img / np.max(img)
    heatmap = cv.applyColorMap(np.uint8(255 * mask), cv.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam
def cam(img_path, model, label_index):
    cv_img = cv.imread(img_path)
    cv_img = cv.resize(cv_img, (224,224))
    in_img = torch.tensor(np.float32(cv.cvtColor(cv_img , cv.COLOR_BGR2RGB)) / 255.)
    
    in_img = in_img.permute(2,0,1).unsqueeze(0)
    hook = Hook(model)
    
    
    pred_vec = hook(in_img)

    # hook.feature torch.Size([1, 128, 28, 28])
    # hook.grad torch.Size([1, 128, 28, 28])
    grad_weight = torch.zeros_like(pred_vec)
    grad_weight[:,label_index] = 1
    pred_vec.backward(grad_weight)
    
    alpha = torch.sum(hook.grad, dim = [-1,-2]) # (-1,128)
    
    mask = F.relu(torch.sum(hook.feature * alpha.view(1,-1,1,1), dim = 1)) # (1,24,24)
    mask = cv.resize(mask.permute(1,2,0).data.numpy(), (224,224))
    mask = mask / np.max(mask)
    show_img = show(cv_img, mask)
    cv.imshow("",show_img)
    cv.waitKey(0)
if __name__ == "__main__":
    img_path = "/home/xueaoru/图片/fish.jpg"
    model = resnet18(pretrained = True)
    cam(img_path, model, 0)
    

    
