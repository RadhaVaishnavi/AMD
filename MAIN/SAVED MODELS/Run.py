import read
import Ext_Feature
import Proposed_PyMFT_Net.run
import ResNet50.Resnet
import CM_CNN.CNN
import VGG16.Vgg
import SVM.SVM

def callmain(tr):
    ACC,TPR,TNR=[],[],[]
    Input_path=read.image('Dataset/test/*/*')

    segmented_Image=read.image('Processed/Segmented_Image/*')
    Fea,Label=Ext_Feature.callmain(segmented_Image)
    #-------------Proposed_PyMFT_Net-------------------
    Proposed_PyMFT_Net.run.callmain(Input_path, Fea, Label, tr, ACC, TPR, TNR)
    #------------Comparative Method-------------
    ResNet50.Resnet.classify(Fea, Label, tr, ACC, TPR, TNR)
    CM_CNN.CNN.Classify(Fea, Label, tr, ACC, TPR, TNR)
    VGG16.Vgg.Classify(Fea, Label, tr, ACC, TPR, TNR)
    SVM.SVM.classify(Fea,Label,tr,ACC,TPR,TNR)

    return ACC,TPR,TNR
