import matplotlib.pyplot  as plt
import PySimpleGUI as sg
sg.change_look_and_feel('LightBlue6')  # look and feel theme
from Main.Features.LTXOR import local_bin_val
from Main.Features.Statistical_Feature import *
from Main.Features.Curvature import curvature_fea
from Main.Features.Reflectivity import Reflectivity
from Proposed_PyMFT_Net.base_model import PyramidNetBuilder
from Proposed_PyMFT_Net.DMO import *

from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from keras.models import Sequential
from CE_NET import cenet
from CE_NET.metrics import dice_coef, dice_loss
from CE_NET.data_generator import *
from skimage.color import rgb2gray
import numpy as np

def mask_to_3d(mask1):
    mask = np.squeeze(mask1)
    mask = [mask, mask, mask]

    return mask,mask1
def generated_mask(Image):

    gray_Image = rgb2gray(Image)

    binarized_gray = (gray_Image > 2.5 * 0.1) * 1  # 2.5
    Img = binarized_gray * 255

    return Img

def cenet_seg(img):
    file_path = "CE_NET/files/"
    model_path = "CE_NET/files/cenet.h5"

    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 500

    cenet.CE_Net_(image_size)
    model = Sequential()

    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    # --------testing-------------------

    ## Generating the result

    image = img
    mask_Img = generated_mask(img)

    predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
    predict_mask = (predict_mask > 0.5) * 255.0

    sep_line = np.ones((image_size, 10, 3)) * 255

    mask,gt = mask_to_3d(predict_mask)
    predict_mask,predicted_output = mask_to_3d(mask_Img)

    return predicted_output


def ltxor(seg_img):
    m, n = seg_img.shape
    #gray_scale = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    lgbp_photo = np.zeros((m, n), np.uint8)
    # converting image to lbp
    for i in range(0, m):
        for j in range(0, n):
            lgbp_photo[i, j] = local_bin_val(seg_img, i, j)
    histograms, bins = np.histogram(lgbp_photo, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    histograms = histograms.flatten()
    return np.mean(histograms)


def statistical_feature(img):
    Mean = img.mean()

    kur = scipy.stats.kurtosis(img)
    kurt=np.nan_to_num(kur[0])

    sk = scipy.stats.skew(img)
    Skew=np.nan_to_num(sk[0])


    glcm = getGlcm(img, 1, 0)
    asm, con, ent, ene = feature_computer(glcm)
    glcm = graycomatrix(img, [5], [0], 256, symmetric=True, normed=True)

    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return Mean,kurt,Skew,ent,correlation,energy

def proposed_classify(file_path, Fea, cls):
    y1=classifier1(file_path,cls)

    y2=classifier2(Fea,cls)
    # -----Applying Taylour series--------------

    y3 = [int(y1[i]) * int(2 + y2[i]) for i in range(len(y2))]
    predict = y3
    return predict,cls
def classifier1(X_test,label):

    nc = len(np.unique(label))
    model = PyramidNetBuilder.build(input_shape=(32, 32, 3), num_outputs=nc, block_type='basic', alpha=240, depth=110,
                                    mode="projection")

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    Initial_weight = 1
    weight = model.get_weights()
    # model.fit(np.array(X_train),np.array(y_train),verbose=0)
    X_test = X_test[0][0:50]
    X_test = np.resize(X_test, (len(X_test), 32, 32, 3))
    pred = model.predict(np.array(X_test))
    weight = np.mean(weight[0]) + Initial_weight
    pred = [pred[i][0] for i in range(len(pred))]

    y1 = int(weight) * pred
    return y1
def classifier2(Fea,label):

    X_test = np.resize(Fea, (len(Fea), 64, 64, 3))
    weight = model2.get_weights()
    weight = np.mean(weight[0]) + Initial_weight
    pred = model2.predict(X_test)
    pred = [pred[i][0] for i in range(len(pred))]

    y = int(weight) * pred
    return y

def label(path):
    ff=str(path).split('/')
    fn=ff[-2]
    if fn=='CNV':Label=1
    elif fn == 'DME':Label=2
    elif fn == 'DRUSEN':Label=3
    elif fn == 'NORMAL':Label=0
    return Label
# Designing layout
#file_path = filedialog.askopenfilename()
layout = [[sg.Text("Choose a file: "), sg.FileBrowse(),sg.Button("OK", size=(10, 2))],
                [ sg.In(key='51', size=(50, 50))]]
# Create the Window layout
window = sg.Window('DemoGUI', layout)

# event loop
while True:
    event, values = window.read()  # displays the window
    if event == 'OK':
        file_path = values['Browse']
        original_image = cv2.imread(file_path)
        plt.title("OCT Image")
        plt.imshow(original_image)
        plt.show()

    seg_Image=cenet_seg(original_image)
    plt.title("Segmented Image")
    plt.imshow(seg_Image)
    plt.show()
    #-----Features---
    f1=curvature_fea(seg_Image)
    f2=Reflectivity(seg_Image)
    f4=ltxor(seg_Image)
    f5,f6,f7,f8,f9,f10=statistical_feature(seg_Image)
    Fea=[f1,f2,f4,f5,f6,f7,f8,f9,f10]

    cls=label(file_path)
    ACC,TPR,TNR=[],[],[]
    pred,classified_output=proposed_classify(original_image, Fea, cls)

    if classified_output==1:A='CNV'
    elif classified_output==2:A='DME'
    elif classified_output==3:A='DRUSEN'
    elif classified_output==0:A='NORMAL'

    window.element('51').Update(A)



