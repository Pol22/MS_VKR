import requests
from PIL import Image
from io import BytesIO
import numpy as np
from queue import Queue
import threading


IMG_SIZE = 96
NUMBER = 350000
HR_DIR = './HR/'
LR_DIR = './LR/'
global_lock = threading.Lock()


def get_and_save(i, queue):
    print('Thread', i, 'started!')
    global NUMBER
    while 1 > 0:
        try:
            url = queue.get()
            response = requests.get(url, timeout=1)
            img = Image.open(BytesIO(response.content))
            width, height = img.size
            if width < IMG_SIZE or height < IMG_SIZE:
                continue
            x = np.random.randint(0, width - IMG_SIZE)
            y = np.random.randint(0, height - IMG_SIZE)
            hr_img = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
            lr_img = hr_img.resize((IMG_SIZE//4, IMG_SIZE//4), Image.BICUBIC)
            name = url.split('/')[-1]
            hr_img.save(HR_DIR + name)
            lr_img.save(LR_DIR + name)
            with global_lock:
                NUMBER = NUMBER - 1
                if NUMBER % 100 == 0:
                    print('Left:', NUMBER)
        except:
            continue

# with open('ImageNet_URLs.txt') as f:
#     while 1 > 0:
#         try:
#             url = f.readline().split()[0]
#             response = requests.get(url, timeout=1)
#             img = Image.open(BytesIO(response.content))
#             width, height = img.size
#             if width < IMG_SIZE or height < IMG_SIZE:
#                 continue
#             x = np.random.randint(0, width - IMG_SIZE)
#             y = np.random.randint(0, height - IMG_SIZE)
#             hr_img = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
#             lr_img = hr_img.resize((IMG_SIZE//4, IMG_SIZE//4), Image.BICUBIC)
#             name = url.split('/')[-1]
#             hr_img.save(HR_DIR + name)
#             lr_img.save(LR_DIR + name)
#             NUMBER = NUMBER - 1
#             if NUMBER == 0:
#                 break
#             if NUMBER % 1000 == 0:
#                 print('Left:', NUMBER)
#         except:
#             continue
        

if __name__ == '__main__':
    queue = Queue()
    for i in range(20):
        thread = threading.Thread(target=get_and_save, args=[i, queue])
        thread.daemon = True
        thread.start()

    
    with open('ImageNet_URLs.txt') as f:
        while 1 > 0:
            if NUMBER <= 0:
                break
            try:
                url = f.readline().split()[0]
                queue.put(url)
            except:
                continue

    
    