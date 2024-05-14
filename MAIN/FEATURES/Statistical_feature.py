from skimage.feature import graycomatrix, graycoprops
import cv2,math
import numpy as np
import scipy
from Main import read
gray_level = 64
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape


    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1
def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape
    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret
def feature_computer(p):
    '''
    :param p: 单张灰度图片矩阵
    :return: 四个 GLCM 特征值
    '''
    Con = 0.0
    Ent = 0.0
    Asm = 0.0
    energy = 0.0

    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            energy += p[i][j] / (1 + (i - j) * (i - j))

            if p[i][j] > 0.0:
                Ent += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Ent, energy
def Stat_Fea(Img_path):
    Mean_,Kurtosis_,Skew_=[],[],[]
    Energy,Entrophy,Correlation=[],[],[]

    for i in range(len(Img_path)):
        print("Stat Fea :", i)
        img = cv2.imread(Img_path[i])

        Mean= img.mean()
        Mean_.append(Mean)

        kur = scipy.stats.kurtosis(img)
        Kurtosis_.append(np.nan_to_num(kur[0]))

        sk = scipy.stats.skew(img)
        Skew_.append(np.nan_to_num(sk[0]))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = getGlcm(img_gray, 1, 0)
        asm, con, ent, ene = feature_computer(glcm)
        glcm = graycomatrix(img_gray, [5], [0], 256, symmetric=True, normed=True)

        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        Energy.append(energy)
        Entrophy.append(ent)
        Correlation.append(correlation)

    Stat_Fea=np.column_stack((Mean_,Kurtosis_,Skew_,Energy,Entrophy,Correlation))


    np.savetxt("Processed/Extracted_Features/Statistical_Fea.csv",Stat_Fea,delimiter=',',fmt='%s')
    return Stat_Fea




