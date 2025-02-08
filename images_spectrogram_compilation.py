import numpy as np
import cv2
from scipy.signal import correlate
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import cv2

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
    """Сдвигает изображение img на shift_x пикселей"""
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted_img

img_path=["./pictures/04000000_1574696114_Raw_0_gray.png",
          "./pictures/04050010_1573836822_Raw_0_gray.png",
          "./pictures/04050029_1574114268_Raw_0_gray.png",
          "./pictures/04050029_1574114625_Raw_0_gray.png",
          "./pictures/04050030_1574193843_Raw_0_gray.png",
          "./pictures/04050030_1574456771_Raw_0_gray.png"]

img=[]
profile=[]

for path in img_path:
    img.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))

for i in img:
    profile.append(np.mean(i, axis=0))


for i in range (0, len(img)-1):
    img1_colored = cv2.applyColorMap(img[i].astype(np.uint8), cv2.COLORMAP_MAGMA)
    img2_colored = cv2.applyColorMap(img[i+1].astype(np.uint8), cv2.COLORMAP_OCEAN)
    alpha= 0.5

    output_path = f"C:/Users/Anasnas/Documents/bakalavr/pictures_diff/diff_{i}.png"
    image = cv2.addWeighted(img1_colored, 1 - alpha, img2_colored, alpha, 0)
    cv2.imwrite(output_path, image)

for i in range (0, len(img)-1):
    best_x_shift = find_best_x_shift(img[i], img[i+1])
    aligned_img2 = shift_image(img[i+1], best_x_shift)
    img1_colored = cv2.applyColorMap(img[i].astype(np.uint8), cv2.COLORMAP_MAGMA)
    img2_colored = cv2.applyColorMap(aligned_img2.astype(np.uint8), cv2.COLORMAP_OCEAN)
    alpha= 0.5

    output_path = f"C:/Users/Anasnas/Documents/bakalavr/pictures_diff/test_diff_{i}.png"
    image = cv2.addWeighted(img1_colored, 1 - alpha, img2_colored, alpha, 0)
    cv2.imwrite(output_path, image)


