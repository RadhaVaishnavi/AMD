import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def Classify(Data,Label,tr,ACC,SEN,SPE):


    x_train, x_test, y_train, y_test = train_test_split(Data, Label, train_size=tr, random_state=0) # 70% for train and 30% for test

    # Defining Model
    # Using Sequential() to build layers one after another
    model = tf.keras.Sequential([

        # Flatten Layer that converts images to 1D array
        tf.keras.layers.Flatten(),

        # Hidden Layer with 512 units and relu activation
        tf.keras.layers.Dense(units=512, activation='relu'),

        # Output Layer with 10 units for 10 classes and softmax activation
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


    # Making Predictions
    predict = model.predict(x_test)

    target = y_test

    tp, tn, fn, fp = 0, 0, 1, 1
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(predict)):

            if target[i][0] == c and predict[i][0] == c:
                fp += 1
            if target[i][0] != c and predict[i][0] != c:
                tn += 1
            if (target[i][0] == c and predict[i][0] != c):
                tp += 1
            if (target[i][0] != c and predict[i][0] == c):
                fn += 1

    acc = (tp + tn) / (tp + fn + tn + fp)
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)

    ACC.append(acc)
    SPE.append(spe)
    SEN.append(sen)





