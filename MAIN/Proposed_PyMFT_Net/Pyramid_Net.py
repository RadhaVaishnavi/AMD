from Proposed_PyMFT_Net.base_model import PyramidNetBuilder
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
def Classify(Imgs,label,tr):

    Data=[]
    for i in range(len(Imgs)):
        Data.append(cv2.imread(Imgs[i]))

    X_train, X_test, y_train, y_test = train_test_split(Data, label, train_size=tr, random_state=42)
    nc=len(np.unique(y_train))
    model=PyramidNetBuilder.build(input_shape=(32,32,3), num_outputs=nc, block_type='basic', alpha=240, depth=110,mode="projection")

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    Initial_weight=1
    weight=model.get_weights()
    #model.fit(np.array(X_train),np.array(y_train),verbose=0)
    X_test=[X_test[i][0][0:50] for i in range(len(X_test))]
    X_test=np.resize(X_test,(len(X_test),32,32,3))
    pred=model.predict(np.array(X_test))
    weight=np.mean(weight[0])+Initial_weight
    pred=[pred[i][0] for i in range(len(pred))]

    y1=int(weight)*pred

    return y1,y_test


