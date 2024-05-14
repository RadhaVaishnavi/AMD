from Proposed_PyMFT_Net import Pyramid_Net
from Proposed_PyMFT_Net import DMO
import numpy as np

def callmain(Input_Img,Fea,Label,tr,ACC,TPR,TNR):

    y1,ytest=Pyramid_Net.Classify(Input_Img,Label,tr)
    y2,ytest=DMO.classify(Fea,Label,tr)

    #-----Applying Taylour series--------------

    y3=[int(y1[i])*int(2+y2[i]) for  i in range(len(y2))]
    predict=y3
    target = ytest

    tp, tn, fn, fp = 1, 1, 1, 1
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(predict)):

            if target[i] == c and predict[i] == c:
                fp += 1
            if target[i] != c and predict[i] != c:
                tn += 1
            if (target[i] == c and predict[i] != c):
                tp += 1
            if (target[i] != c and predict[i] == c):
                fn += 1

    Acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    ACC.append(Acc)
    TPR.append(tpr)
    TNR.append(tnr)
    return Label
