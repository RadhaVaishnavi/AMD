from Main.Features import Reflectivity
from Main.Features import Thickness
from Main.Features import Curvature
from Main.Features import Statistical_Feature
from Main.Features import LTXOR
import pandas as pd
import numpy as np
import cv2,glob

def fea(segmented_Images):
    Statistical_Feature.Stat_Fea(segmented_Images)
    LTXOR.ltxor(segmented_Images)

    F=[]
    for i in range(len(segmented_Images)):
        print('i :',i)
        img=cv2.imread(segmented_Images[i],0)
        f1=Reflectivity.Reflectivity(img)
        f2=Thickness.thickness_calc(img)
        f3=Curvature.curvature_fea(img)
        f=[f1,f2,f3]
        F.append(f)
        np.savetxt("Processed/Extracted_Features/Fea1.csv",F,delimiter=',',fmt='%s')

def callmain(Seg_Img):

    #fea(Seg_Img)

    pa = 'Processed/Extracted_Features/*'

    # setting the path for joining multiple files
    data = []
    # list of merged files returned
    files = glob.glob(pa)

    for file in files:
        df_list = pd.read_csv(file, header=None)
        data.append(df_list)
    # joining files with concat and read_csv

    feature = pd.concat(data, axis=1)
    feature = np.array(feature)

    Label=pd.read_csv("Processed/Label.csv",header=None)
    Label=np.array(Label)

    return feature,Label
