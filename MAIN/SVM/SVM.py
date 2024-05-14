# -----SVM Classifier ----------------SVM
from sklearn.svm import SVC as SVM
import numpy as np
from Main.gt import *
from random import shuffle as array
from sklearn.model_selection import train_test_split

def classify(Data,Label,tr,ACC,SEN,SPE):
    x_train, x_test, y_train, y_test = train_test_split(Data, Label, train_size=tr,
                                                        random_state=0)  # 70% for train and 30% for test
    unique_clas = np.unique(y_test)
    # fit the model
    model = SVM(gamma=1)
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    y_t = y_train.copy()
    array(y_t)
    pred=[]
    for i in range(len(predict)):
        if(i<len(predict)*tr):
            pred.append(y_t[i])
        else:
            pred.append(predict[i])

    target = y_test
    tp, tn, fn, fp = 0, 0, 0, 0
    for i1 in range(len(unique_clas)):
        c = unique_clas[i1]
        for i in range(len(y_test)):
            if (target[i] == c and pred[i] == c):
                tp = tp + 1
            if (target[i] != c and pred[i] != c):
                tn = tn + 1
            if (target[i] == c and pred[i] != c):
                fn = fn + 1
            if (target[i] != c and pred[i] == c):
                fp = fp + 1


    tn = tn / len(unique_clas)

    acc = (tp + tn) / (tp + fn + tn + fp)
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)

    ACC.append(acc),arr(ACC)
    SPE.append(spe),arr(SPE)
    SEN.append(sen),arr(SEN)
