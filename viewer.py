from matplotlib.mlab import window_hanning,specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np

from PIL import Image

import io
import scipy
from scipy import signal
from scipy.signal import butter, lfilter

import requests


fig = plt.figure(1)
plt.axis('off')

def update_fig(n):
    try:
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}
        data = open('source.jpg', 'rb').read()
        response = requests.post(test_url, data=data)
        image_data = response.content
        art_image = Image.open(io.BytesIO(image_data))
        
        im_art.set_data(art_image)
        print(n)
        return im_art
    except:
        pass


addr = 'http://iccluster026.iccluster.epfl.ch:5001'
test_url = addr + '/api/style/rain_princess'

    
content_type = 'image/jpeg'
headers = {'content-type': content_type}
data = open('source.jpg', 'rb').read()
response = requests.post(test_url, data=data)
image_data = response.content
art_image = Image.open(io.BytesIO(image_data))
im_art = plt.imshow(art_image)
    
anim = animation.FuncAnimation(fig,update_fig,blit = False, interval=1000)
plt.show()

    
    
