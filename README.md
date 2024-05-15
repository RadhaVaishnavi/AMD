# AMD
# Early diagnosis and grading of age related macular degeneration using deep learning 
![IMG_0283](https://github.com/RadhaVaishnavi/AMD/assets/84319477/b56b2a6d-8f35-4c9a-ac4e-13434b55e8a2)

One particular use of machine and deep learning is classification. In this thesis we 
propose a novel classification model based on deep learning that will be able to classify and
identify different retinal diseases at early stages with high accuracy that outperform the state
of the art approaches, previous works and the diagnosis of experts in ophthalmology. Vision
and eye health are one the most crucial things in human life, it needs to be preserved to maintain
life of the individuals. Eye diseases such as CNV, DRUSEN, AMD, DME are mainly caused
due to the damages of the retina, and since it is damaged, and diagnosed at late stages, there is
almost no chance to reverse vision and cure it, which means that the patient will loose the
vision ability partially, and may be entirely. We propose here three deep convolutional neural
networks architectures and tuned three pretrained models that have been trained on large
dataset such as Inception, VGG-16 and ResNET. In order to evaluate our model we have
compared the diagnosis of 7 experts with the performence of our model. Our proposed model
outperformed the diagnosis of 6 out of 7 experts also the previous published works using the
same data, where we obtained an accuracy up to 99.27%.

# Designed PyMFT-Net model for AMD detection
This project presents PyMFT-Net designed for the detection of AMD by utilizing OCT images. At first, the OCT image accumulated from the database is allowed for layer segmentation using CE-Net [9] with modified loss functions formulated using Dice, Tversky loss, weighted binary cross-entropy, and regularization loss. Then, the extraction of features, like reflectivity, thickness, curvature [10], statistical features like mean, correlation, energy, skewness, entropy, and kurtosis [27], and the Local Texton XOR pattern (LTXOR) [11] is executed using different feature extractors. Later, the AMD detection is executed using the PyMFT-Net approach. Here, the PyMFT-Net is designed by incorporating PyramidNet [12], DMN [13], and Taylor series [14]. Finally, the output obtained while detecting AMD is categorized into four types, namely CNV, DME, DRUSEN, and normal. In addition, figure 1 shows the systematic view of PyMFT-Net technique designed for AMD detection.
![Screenshot 2024-05-15 134458](https://github.com/RadhaVaishnavi/AMD/assets/84319477/c9e1ff5f-f580-4bd3-8e11-ebe339867ce7)
Systematic view of PyMFT-Net technique used for the detection of AMD

# Acquisition of input image
The input image is initially accumulated from OCT and Chest X-Ray datasets [15] for the detection of AMD and the dataset is given as,
A  A1, A2 , A3,..., AR..., AW 
Here, the database accumulated for the detection of AMD is symbolized as A , the Rth OCT
image taken into consideration to detect AMD is given by A , and W represents total data available in the dataset.

# Layer segmentation using CE-Net
 Generally, the layer segmentation method helps to effectively distinguish various retinal layers to assist clinicians in monitoring and identifying eye diseases, namely AMD, diabetic retinopathy, glaucoma, and so on. The spatial smoothness as well as continuity of healthy retinal layers are identified by layer segmentation, thereby determining the changes in
retinal layers due to disease. In this research, the input OCT image  is initially subjected to
R CE-Net [9] along with modified loss functions for the segmentation of retinal layers. Here, the loss functions, like Dice, Tversky loss, weighted binary cross-entropy, and regularization loss are considered for the segmentation task. The segmentation of layer from the input OCT image A is explicated as follows,
# Architecture of CE-Net
The CE-Net [9] uses a Residual Multi-kernel Pooling (RMP) block and Dense Atrous Convolution (DAC) block for preserving the spatial information and get more abstract features to perform the segmentation task. The CE-Net mainly possesses three layers, namely the feature decoder module, feature context extractor module, and feature encoder module. The processes carried out are demonstrated below,
(a)	Feature encoder module: In this layer, pretrained ResNet is used instead of the encoder module in the U-Net while retaining the blocks used for feature extraction without the fully connected and average pooling layers. Moreover, it also accelerates the network convergence and avoids vanishing gradient issues by adding a shortcut mechanism.

(b)	Feature context extractor module: The context extractor module is used for the extraction of high-level feature maps, which also extract context semantic information from the input image. This module mainly comprises the RMP block and DAC block, which also adopt atrous convolution to execute the segmentation task. Due to pooling layers, the atrous convolution is utilized to perform dense segmentation for overcoming semantic information loss in images. The atrous convolution is performed to effectively determine the wavelet transform of two- dimensional signals. Similarly, the DAC blocks are used to encode high-level semantic feature
maps, which possess four cascade branches with increasing total atrous convolution. Then, DAC uses various receptive fields and applies convolution for Rectified Linear Units (ReLU) activation in each atrous branch. At last, the actual features are directly added with other features, like the ResNet shortcut mechanism. The DAC block is used for the extraction of features from objects with different sizes by incorporating atrous convolution with various atrous rates.
Generally, large variations in object size pose a major challenge in image segmentation. Thus, RMP blocks that rely on various effective field-of-views are employed to identify objects of various sizes. The global context information with various receptive field sizes is encoded by the RMP. A convolution is performed after each level of pooling to decrease the dimension of weights and computational cost. Further, the upsampling of the low-dimension feature map is executed to get the features of the same size as the actual feature map using bilinear interpolation and the upsampled original feature maps are finally concatenated.

(c)	Feature decoder module: The extracted large semantic features from the feature extractor module and context extractor module are restored by adopting the feature decoder module. Here, the information loss is remedied because of striding convolution operation and consecutive pooling by taking some detailed information from the encoder to the decoder by using skip connections. Moreover, the decoding performance is increased by adopting an efficient block. The decoder performs deconvolution and upscaling operations. Here, the deconvolution operation enlarges the image, and the upscaling operation is performed to enhance the image size using linear interpolation. Particularly, transposed convolution is used to learn self-adaptive mapping for restoring more detailed information features. Finally, the resultant segmented output ES is obtained from the feature decoder module based on the decoder block and skip connections from the input OCT image AR . In addition, figure 2 elucidates the structure of CE-Net for the segmentation of layers.
![Screenshot 2024-05-15 135150](https://github.com/RadhaVaishnavi/AMD/assets/84319477/50580fbf-a269-4fea-9181-d53cc6b53073)

# DATASET & IMPLEMENTATION DETAILS

# Dataset description
We have used the OCT images dataset from, the dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category. There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN). Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.
# Image Labeling
Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing expertise f or verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduateand medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram. To account for human error in grading, a validationsubset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.
# Dataset pre-processing
Preprocessing phase is one crucial phase in image classification tasks, images needs to be looked at, the shape, the size and the balanced class are preliminary things. Our dataset was taken from different research labs, what makes the sizes of the images very different (496, 768, 3), (496, 1024, 3), (496, 512, 3), (496, 1536, 3), (512, 512, 3) the first two values refers to the width and the height of the image, and the third one refers to the image channels , meaning in this case that the images are in RGB 1 .

![Screenshot 2024-05-15 135451](https://github.com/RadhaVaishnavi/AMD/assets/84319477/91a0b29b-5ee0-437d-a00f-5eada630fa52)

Figure 4.1: Plotting the images on each different channel

•	Images resizing: Importing the images with the original sizes will lead to use big part of hardware resources and the time of processing will highly increase, we decided to reduce the image sizes to 224 X 224 pixels, same as Imagenet dataset images sizes.

•	Data Resampling: The dataset is splitted into 3 folders as explained in the previous section, with only 8 images per class for validation and 242 images for test and the rest for training. as detailed in table URL 3.1. This split is not efficient and can lead to extream overfitting. We made another split of 80% for training , 20% for validation (table 3.2) and after constructing our model we tested our model on 968 images.

•	Balanced classes The dataset is unbalanced, meaning each class have different number of images, which can lead to overfitting in some cases, we decided to do experiments on both balanced and unbalanced classes cases.

![Screenshot 2024-05-15 135718](https://github.com/RadhaVaishnavi/AMD/assets/84319477/aa39607b-297d-4fda-84e7-c7d0801e8115)
![Screenshot 2024-05-15 135728](https://github.com/RadhaVaishnavi/AMD/assets/84319477/8211afab-1ff9-4081-a849-ebc6b5748c43)

Encoding categories The class labels are strings ( CNV , DRUSEN,NORMAL, DME) Since we are working with Numpy python library, it only accepts array lists numbers, we need to ’one-hot-encode’ our target variable. One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in . This means that a column will be created for each output category and a binary variable is inputted for each category.

![Screenshot 2024-05-15 135817](https://github.com/RadhaVaishnavi/AMD/assets/84319477/4031a471-b951-40c7-8cdb-0a450af4f7e4)

For example, the CNV category in the dataset is a 1. This means that the first number in our array will have a 1 and the rest of the array will be filled with 0.

![Screenshot 2024-05-15 135902](https://github.com/RadhaVaishnavi/AMD/assets/84319477/011dcf95-0519-4588-ac09-8b6b5ddc705f)

# Comparative discussion
The table 5.1 represents the proposed PyMFT-Net techniques with implemented techniques with TNR, TPR, and accuracy. The accuracy of 91.876% is attained by the PyMFT-Net while the accuracy evaluated by CM-CNN, SVM, VGG-16, and ResNet50 are 84.765%, 85.78676%, 88.765%, and 89.5435 correspondingly for learning set of 90%,. The high TNR value of 92.6545% is produced by PyMFT-Net, while the TNR evaluated by CM-CNN, SVM, VGG-16, and ResNet50 are 85.876%, 86.755%, 87.5434%, and 89.7554%. The maximum TPR of 94.7866% is achieved by the devised PyMFT-Net, while the implemented techniques CM-CNN, SVM, VGG-16, and ResNet50 are 88.8665%, 89.53435%, 90.5343%, and 91.765% correspondingly. Massive enhancement in performance is seen in the proposed PyMFT-Net technique because of the utilization of the PyramidNet model and the application of DMN for AMD detection. The PyramidNet has the ability to decrease the feature map dimensionality while improving the efficacy and the DMN is highly effective in resource-constrained environments and thus the fusion of these two networks led to enhanced detection performance.
![Screenshot 2024-05-15 140039](https://github.com/RadhaVaishnavi/AMD/assets/84319477/2b6cdf0a-5c98-474b-b9fb-ac79a5f98ab2)

# OUTPUTS

![Screenshot 2024-05-15 141123](https://github.com/RadhaVaishnavi/AMD/assets/84319477/07eefc3f-f662-4ca6-8421-80f86cd1e86b)

![Screenshot 2024-05-15 141525](https://github.com/RadhaVaishnavi/AMD/assets/84319477/b081dcc9-0d6c-4725-b38b-35f895b9525d)

![Screenshot 2024-05-15 140745](https://github.com/RadhaVaishnavi/AMD/assets/84319477/8d1942e7-81ee-48ec-a168-8d0e9354eeec)
![Screenshot 2024-05-15 141537](https://github.com/RadhaVaishnavi/AMD/assets/84319477/e90738e5-9c42-4d2e-aef7-3eade6528831)
![Screenshot 2024-05-15 141548](https://github.com/RadhaVaishnavi/AMD/assets/84319477/4b3fe551-221c-41f1-9d67-87d8fb1ac2b5)

![Screenshot 2024-05-15 141433](https://github.com/RadhaVaishnavi/AMD/assets/84319477/dc2f44df-f902-457e-b89b-35bcef8c1c02)
![Screenshot 2024-05-15 141316](https://github.com/RadhaVaishnavi/AMD/assets/84319477/e1c6daf8-0953-4a76-bdce-f5fe72155cc8)
![Screenshot 2024-05-15 141325](https://github.com/RadhaVaishnavi/AMD/assets/84319477/02ee40e4-f2df-4bdc-8f45-71522d2b919e)
![Screenshot 2024-05-15 141334](https://github.com/RadhaVaishnavi/AMD/assets/84319477/00230963-5731-4af1-b589-6e64e09ac6fc)
![Screenshot 2024-05-15 141342](https://github.com/RadhaVaishnavi/AMD/assets/84319477/88159ffd-c5eb-4064-8311-ff96a9bf473f)
![Screenshot 2024-05-15 141352](https://github.com/RadhaVaishnavi/AMD/assets/84319477/dfd11ab9-0cff-4cf1-857c-dcd92365fa24)
![Screenshot 2024-05-15 141400](https://github.com/RadhaVaishnavi/AMD/assets/84319477/834e0e23-edf2-41a1-98e4-686752e828fc)
![Screenshot 2024-05-15 141408](https://github.com/RadhaVaishnavi/AMD/assets/84319477/a854758d-48df-4d60-b88b-32046f6731a9)
![Screenshot 2024-05-15 141416](https://github.com/RadhaVaishnavi/AMD/assets/84319477/f3a54502-35e2-4b14-a018-40b8dbb303de)
![Screenshot 2024-05-15 141425](https://github.com/RadhaVaishnavi/AMD/assets/84319477/6c684cf0-4f4f-4629-8f36-a88251e9f8f7)
![Screenshot 2024-05-15 141449](https://github.com/RadhaVaishnavi/AMD/assets/84319477/c27eed47-c0a5-4ed9-9410-c97d88cc0ef9)
![Screenshot 2024-05-15 141441](https://github.com/RadhaVaishnavi/AMD/assets/84319477/6683dd5c-d041-4132-bf96-7f3e2ab3fe96)






