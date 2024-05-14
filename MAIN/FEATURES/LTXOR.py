import cv2,glob,re
import numpy as np


def assign_bit(picture, x, y, c):  # comparing bit with threshold value of centre pixel
    bit = 0
    try:
        if picture[x][y] >= c:
            bit = 1
    except:
        pass
    return bit


def local_bin_val(picture, x, y):  # calculating local binary pattern value of a pixel
    eight_bit_binary = []
    centre = picture[x][y]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    decimal_val = 0
    # starting from top right,assigning bit to pixels clockwise
    eight_bit_binary.append(assign_bit(picture, x - 1, y + 1, centre))
    eight_bit_binary.append(assign_bit(picture, x, y + 1, centre))
    eight_bit_binary.append(assign_bit(picture, x + 1, y + 1, centre))
    eight_bit_binary.append(assign_bit(picture, x + 1, y, centre))
    eight_bit_binary.append(assign_bit(picture, x + 1, y - 1, centre))
    eight_bit_binary.append(assign_bit(picture, x, y - 1, centre))
    eight_bit_binary.append(assign_bit(picture, x - 1, y - 1, centre))
    eight_bit_binary.append(assign_bit(picture, x - 1, y, centre))
    # calculating decimal value of the 8-bit binary number
    for i in range(len(eight_bit_binary)):
        decimal_val += eight_bit_binary[i] * powers[i]
    return decimal_val

def ltxor(Img_list):

    LTXOR_F = []
    for ii in range(len(Img_list)):
        print("LTXOR :", ii)
        photo = cv2.imread(Img_list[ii])
        m, n, _ = photo.shape
        gray_scale = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
        lgbp_photo = np.zeros((m, n), np.uint8)
        # converting image to lbp
        for i in range(0, m):
            for j in range(0, n):
                lgbp_photo[i, j] = local_bin_val(gray_scale, i, j)
        histograms, bins = np.histogram(lgbp_photo, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        histograms = histograms.flatten()
        LTXOR_F.append(histograms)
        np.savetxt("Processed/Extracted_Features/LTXOR_Features.csv",LTXOR_F, delimiter=',', fmt='%s')
    return LTXOR_F
