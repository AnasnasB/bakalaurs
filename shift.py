import numpy as np
import cv2
from scipy.signal import correlate
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os

def find_best_x_shift(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    max_width = min(img1_gray.shape[1], img2_gray.shape[1])
    best_shift = 0
    best_corr = -1

    for shift in range(-max_width // 2, max_width // 2):

        M = np.float32([[1, 0, shift], [0, 1, 0]])
        img2_shifted = cv2.warpAffine(img2_gray, M, (img2_gray.shape[1], img2_gray.shape[0]))


        correlation = np.corrcoef(img1_gray.flatten(), img2_shifted.flatten())[0, 1]

        if correlation > best_corr:
            best_corr = correlation
            best_shift = shift

    return best_shift

def shift_image(img, shift_x):
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted_img

path="./pictures"
pictures_path=[]
for file in os.listdir(f"{path}"):
    if file.endswith('_gray.png'):
        pictures_path.append(f"{path}/{file}")

pic=cv2.imread(pictures_path[0], cv2.IMREAD_UNCHANGED)
cv2.imwrite(f"./pictures_with_shift/shifted_picture_gray_0.png", pic)
for i in range(1,len(pictures_path)):
    p=cv2.imread(pictures_path[i], cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f"./pictures_with_shift/shifted_picture_gray_{i}.png", shift_image(p, find_best_x_shift(pic,p)))
    print(i)