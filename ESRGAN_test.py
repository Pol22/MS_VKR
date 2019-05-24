import sys
import os.path
import glob
from PIL import Image
import numpy as np
import torch
import torch.onnx
from torchviz import make_dot
import architecture as arch

model_path = 'RRDB_ESRGAN_x4.pth'
psnr_path = 'models/RRDB_PSNR_x4.pth'
device = torch.device('cpu')

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                      mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
model.train(False)

img = Image.open('SR_test_img/test_lr.png')
# 128 x 128
img = img.resize((128, 128), Image.BICUBIC)
img.show()
img = np.asarray(img, dtype=np.float)
img = img / 255.0
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
img_LR = img.unsqueeze(0)
img_LR = img_LR.to(device)
output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
out = Image.fromarray(np.uint8(output))
out.show()
