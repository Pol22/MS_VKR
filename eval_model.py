import keras
from keras.models import load_model
import numpy as np
from PIL import Image


IMG_SIZE = 96

model_file = 'SRGAN.h5'
img_file = 'recherche-exotiques.jpg'

img = Image.open(img_file)
img = img.crop((400, 10, 400 + IMG_SIZE * 3, 10 + IMG_SIZE * 3))
img.save('test_hr.png')
lr_img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
lr_img.save('test_lr.png')
            
model = load_model(model_file, compile=False)
data = np.asarray(lr_img, dtype=np.float)
data = (data - 127.5) / 127.5
data = np.expand_dims(data, axis=0)
result = model.predict(data)
result = np.squeeze(result)
result = (result + 1) * 127.5
img_arr = np.uint8(result)

sr_img = Image.fromarray(img_arr)
sr_img.save('test_sr.png')