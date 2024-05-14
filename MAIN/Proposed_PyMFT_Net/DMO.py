import numpy as np
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.constraints import max_norm
from sklearn.model_selection import train_test_split



def maxout_activation_function(inputs,units,axis=None):
    if axis is None:
        axis=-1
    input_shape=inputs.get_shape().as_list()
    n_dims=len(input_shape)
    assert n_dims==4
    num_channels=input_shape[axis]
    if num_channels%units :
        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, units))
    input_shape[axis]=units
    input_shape+=[num_channels//units]
    output=K.reshape(inputs,(-1,input_shape[1],input_shape[2],input_shape[3],input_shape[4]))
    output_max=K.max(output,axis=-1,keepdims=False)
    return output_max


# MAking a second model where we use maxout activation function
max_norm = max_norm(max_value=8, axis=[0, 1, 2])
img_shape=(64,64,3)

inp=L.Input(img_shape)
conv1=L.Conv2D(filters=64,kernel_size=(3,3),activation=None, kernel_constraint=max_norm)(inp)
maxout1=L.Lambda(maxout_activation_function,arguments={'units':32})(conv1)
batch1=L.BatchNormalization(momentum=0.8)(maxout1)
pool1 = L.MaxPooling2D(pool_size=(2,2))(batch1)
drop1= L.Dropout(0.6)(pool1)
conv2=L.Conv2D(filters=128,kernel_size=(3,3),activation=None, kernel_constraint=max_norm)(drop1)
maxout2=L.Lambda(maxout_activation_function,arguments={'units':64})(conv2)
batch2=L.BatchNormalization(momentum=0.8)(maxout2)
pool2 = L.MaxPooling2D(pool_size=(2,2))(batch2)
drop2= L.Dropout(0.5)(pool2)
conv3=L.Conv2D(filters=256,kernel_size=(3,3),activation=None, kernel_constraint=max_norm)(drop2)
maxout3=L.Lambda(maxout_activation_function,arguments={'units':64})(conv3)
batch3=L.BatchNormalization(momentum=0.8)(maxout3)
pool3 = L.MaxPooling2D(pool_size=(2,2))(batch3)
drop3= L.Dropout(0.4)(pool3)
flatten=L.Flatten()(drop3)
dense=L.Dense(2,activation='softmax')(flatten)


model2=M.Model(inputs=inp,outputs=dense)
Initial_weight=1


model2.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
def classify(Data,label,tr):


    X_train, X_test, y_train, y_test = train_test_split(Data,label, train_size = tr, random_state = 42)

    xt = len(X_train)
    xt1 = len(X_test)
    X_train = np.resize(X_train, (xt, 64, 64, 3))
    X_test = np.resize(X_test, (xt1, 64, 64, 3))
    y_test = np.resize(y_test, (xt1, 64, 64, 3))
    y_trainx = np.resize(y_train, (xt, 2))
    model2.fit(X_train, y_trainx, epochs=2, batch_size=10, verbose=0)

    weight=model2.get_weights()
    weight = np.mean(weight[0]) + Initial_weight
    pred = model2.predict(X_test)
    pred = [pred[i][0] for i in range(len(pred))]

    y=int(weight)*pred

    return y,y_test









