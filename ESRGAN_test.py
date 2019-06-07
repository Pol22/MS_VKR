from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import architecture as arch
from bagnets.bagnets import bagnet33
from torchsummary import summary

# esrgan_path = 'RRDB_ESRGAN_x4.pth'
# psnr_path = 'RRDB_PSNR_x4.pth'
# device = torch.device('cpu')

# model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None,
#                       act_type='leakyrelu', mode='CNA', res_scale=1,
#                       upsample_mode='upconv')

# model.load_state_dict(torch.load(psnr_path))
# model.eval()
# for k, v in model.named_parameters():
#     v.requires_grad = False
# model = model.to(device)
# model.train(False)

# img = Image.open('SR_test_img/test_lr.png')
# # 128 x 128
# img = img.resize((512, 512), Image.BICUBIC)
# img = np.asarray(img, dtype=np.float)
# img = img / 255.0
# img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
# img_LR = img.unsqueeze(0)
# img_LR = img_LR.to(device)
# output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
# output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
# output = (output * 255.0).round()
# out = Image.fromarray(np.uint8(output))
# out.show()

bagnet = bagnet33(pretrained=True)

bagnet_input_shape = (512, 512)

img = Image.open('SR_test_img/fruit_lr.jpeg')
original = img.resize(bagnet_input_shape, Image.BICUBIC)
sample = np.asarray(original, dtype=np.float) / 255.
sample = [sample, sample]
# Mean and std
sample -= np.array([0.485, 0.456, 0.406])
sample /= np.array([0.229, 0.224, 0.225])

# input_data = torch.from_numpy(np.transpose(sample[:, :, [2, 1, 0]], (2, 0, 1))).float()
# input_data = input_data.unsqueeze(0)
input_data = torch.from_numpy(np.transpose(sample, (0, 3, 1, 2))).float()
print(input_data.shape)

bagnet_crop = nn.Sequential(*list(bagnet.children())[:3])
# summary(bagnet_crop, (3, 224, 224))
# out = bagnet_crop(input_data)


def bagnet_loss(sr_imgs, hr_imgs):
    # Input (batch, 3, 512, 512)
    global bagnet_crop
    sr_out = bagnet_crop(sr_imgs)
    hr_out = bagnet_crop(hr_imgs)
    sub = torch.sub(sr_out, hr_out)
    sq_sub = torch.pow(sub, 2)
    loss = torch.sum(sq_sub, dim=3)
    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    loss = loss / (510 * 510 * 64)
    return torch.mean(loss)

print(bagnet_loss(input_data, torch.from_numpy(np.zeros((1, 3, 512, 512))).float()))
