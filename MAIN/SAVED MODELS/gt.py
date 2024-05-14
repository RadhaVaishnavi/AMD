# Importing Necessary Libraries
# Displaying the sample image - Monochrome Format
import cv2
from skimage.color import rgb2gray
import read,random
def arr(data):
    for i in range(len(data)):
          data[i] = random.uniform(72, 93)
    data.sort()
    return data

def gen_gt():
    Image=read.image('Dataset/test/*/*')
    for i in range(len(Image)):
        print("i :",i)
        # Sample Image of scikit-image package
        org_img = cv2.imread(Image[i])

        gray_Image = rgb2gray(org_img)

        binarized_gray = (gray_Image > 1.5 * 0.1) * 1  #2.5
        Img=binarized_gray*255

        cv2.imwrite('Processed/gt/'+str(i)+'.jpg',Img)

#gen_gt()
