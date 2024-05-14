import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split  # Import train_test_split function
import numpy as np,math



def Classify(Data,Label,tr,ACC,SEN,SPE):

    X_train, X_test, y_train, y_test = train_test_split(Data, Label, train_size=tr, random_state=42)

    batch_size = 64
    epochs = 20
    num_classes = 10
    Nf,Nf_=15,4

    xt=len(X_test)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(num_classes, activation='softmax'))
    X_test = np.resize(X_test, (xt, 28, 28, 1))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',
                          metrics=['accuracy'])
    pred = model.predict(X_test)

    predict = []
    for i in range(len(y_train)): predict.append(y_train[i])
    for i in range(len(pred)):
        if i == 0:
            predict.append(np.argmax(pred[i]))
        else:
            tem = []
            for j in range(len(pred[i])):
                tem.append(np.abs(pred[i][j] - pred[i - 1][j]))
            predict.append(np.argmax(tem))


    target = y_test

    tp, tn, fn, fp = 0, 0, 1, 1
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):
            if target[i] == c and predict[i] == c:
                fp += 1
            if target[i] != c and predict[i] != c:
                tn += 1
            if (target[i] == c and predict[i] != c):
                tp += 1
            if (target[i] != c and predict[i] == c):
                fn += 1

    acc = (tp + tn) / (tp + fn + tn + fp)
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)

    ACC.append(acc)
    SPE.append(spe)
    SEN.append(sen)
