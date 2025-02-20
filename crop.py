import numpy as np
import cv2
from scipy.signal import correlate
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os

path="./pictures_with_shift"
pictures_path=[]
for file in os.listdir(f"{path}"):
    if file.endswith('.png'):
        pictures_path.append(f"{path}/{file}")

i=0
for file in pictures_path:
    image = cv2.imread(file)
    cropped_image = image[0:813, 131:1181]
    cv2.imwrite(f"./pictures_with_shift_crop/shifted_picture_gray_{i}.png", cropped_image)
    i+=1
