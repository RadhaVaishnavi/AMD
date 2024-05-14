import read
import numpy as np

def get_class():
    Input_path = read.image('Dataset/test/*/*')
    Label=[]
    for i in range(len(Input_path)):
        print("i :",i)
        fn=Input_path[i].split('\\')
        fn=fn[1]
        if fn =='CNV':Label.append(1)
        elif fn == 'DME':Label.append(2)
        elif fn == 'DRUSEN':Label.append(3)
        elif fn == 'NORMAL':Label.append(0)
    np.savetxt("Processed/Label.csv",Label,delimiter=',',fmt='%s')

    return Label


#get_class()
