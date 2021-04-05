# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:18:42 2021

@author: prajw
"""

import os
import tensorflow as tf

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

from utils import *
    
image_path = 'img.jpg'
style_path = 'Style.jpg'


content_image = load_img(image_path)
style_image = load_img(style_path)


import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
img = tensor_to_image(stylized_image)
img.save('examples/Styled.jpg')


