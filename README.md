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

3.1.	Designed PyMFT-Net model for AMD detection
This project presents PyMFT-Net designed for the detection of AMD by utilizing OCT images. At first, the OCT image accumulated from the database is allowed for layer segmentation using CE-Net [9] with modified loss functions formulated using Dice, Tversky loss, weighted binary cross-entropy, and regularization loss. Then, the extraction of features, like reflectivity, thickness, curvature [10], statistical features like mean, correlation, energy, skewness, entropy, and kurtosis [27], and the Local Texton XOR pattern (LTXOR) [11] is executed using different feature extractors. Later, the AMD detection is executed using the PyMFT-Net approach. Here, the PyMFT-Net is designed by incorporating PyramidNet [12], DMN [13], and Taylor series [14]. Finally, the output obtained while detecting AMD is categorized into four types, namely CNV, DME, DRUSEN, and normal. In addition, figure 1 shows the systematic view of PyMFT-Net technique designed for AMD detection.


Figure 3.1. Systematic view of PyMFT-Net technique used for the detection of AMD

3.2.	Acquisition of input image
The input image is initially accumulated from OCT and Chest X-Ray datasets [15] for the detection of AMD and the dataset is given as,
A  A1, A2 , A3,..., AR..., AW 	(1)
 
Here, the database accumulated for the detection of AMD is symbolized as A , the Rth OCT
 
image taken into consideration to detect AMD is given by available in the dataset.
3.3.	Layer segmentation using CE-Net
 
A , and W represents total data
 

Generally, the layer segmentation method helps to effectively distinguish various retinal layers to assist clinicians in monitoring and identifying eye diseases, namely AMD, diabetic retinopathy, glaucoma, and so on. The spatial smoothness as well as continuity of healthy retinal layers are identified by layer segmentation, thereby determining the changes in
retinal layers due to disease. In this research, the input OCT image  is initially subjected to
R

CE-Net [9] along with modified loss functions for the segmentation of retinal layers. Here, the loss functions, like Dice, Tversky loss, weighted binary cross-entropy, and regularization loss are considered for the segmentation task. The segmentation of layer from the input OCT image A is explicated as follows,
-Architecture of CE-Net

The CE-Net [9] uses a Residual Multi-kernel Pooling (RMP) block and Dense Atrous Convolution (DAC) block for preserving the spatial information and get more abstract features to perform the segmentation task. The CE-Net mainly possesses three layers, namely the feature decoder module, feature context extractor module, and feature encoder module. The processes carried out are demonstrated below,
(a)	Feature encoder module: In this layer, pretrained ResNet is used instead of the encoder module in the U-Net while retaining the blocks used for feature extraction without the fully connected and average pooling layers. Moreover, it also accelerates the network convergence and avoids vanishing gradient issues by adding a shortcut mechanism.
(b)	Feature context extractor module: The context extractor module is used for the extraction of high-level feature maps, which also extract context semantic information from the input image. This module mainly comprises the RMP block and DAC block, which also adopt atrous convolution to execute the segmentation task. Due to pooling layers, the atrous convolution is utilized to perform dense segmentation for overcoming semantic information loss in images. The atrous convolution is performed to effectively determine the wavelet transform of two- dimensional signals. Similarly, the DAC blocks are used to encode high-level semantic feature
 
maps, which possess four cascade branches with increasing total atrous convolution. Then, DAC uses various receptive fields and applies convolution for Rectified Linear Units (ReLU) activation in each atrous branch. At last, the actual features are directly added with other features, like the ResNet shortcut mechanism. The DAC block is used for the extraction of features from objects with different sizes by incorporating atrous convolution with various atrous rates.
Generally, large variations in object size pose a major challenge in image segmentation. Thus, RMP blocks that rely on various effective field-of-views are employed to identify objects of various sizes. The global context information with various receptive field sizes is encoded by the RMP. A convolution is performed after each level of pooling to decrease the dimension of weights and computational cost. Further, the upsampling of the low-dimension feature map is executed to get the features of the same size as the actual feature map using bilinear interpolation and the upsampled original feature maps are finally concatenated.
(c)	Feature decoder module: The extracted large semantic features from the feature extractor module and context extractor module are restored by adopting the feature decoder module. Here, the information loss is remedied because of striding convolution operation and consecutive pooling by taking some detailed information from the encoder to the decoder by using skip connections. Moreover, the decoding performance is increased by adopting an efficient block. The decoder performs deconvolution and upscaling operations. Here, the deconvolution operation enlarges the image, and the upscaling operation is performed to enhance the image size using linear interpolation. Particularly, transposed convolution is used to learn self-adaptive mapping for restoring more detailed information features. Finally, the resultant segmented output ES is obtained from the feature decoder module based on the decoder block and skip connections from the input OCT image AR . In addition, figure 2 elucidates the structure of CE-Net for the segmentation of layers.
 

 
Input OCT image,
Segmented OCT image,  

Figure 2. Structure of CE-Net
 
Modified loss function

The CE-Net segments each pixel as background or foreground, that is a pixel-wise classification issue. The widely utilized loss functions, namely Dice, Tversky loss, weighted binary cross-entropy, and regularization loss are considered to replace the common loss that is used during the segmentation task. The loss function is expressed as,
Loss function  e1 W   e2  X   e3 Y   e4 Z 	(2)

where, the scaling factors are signified as e , e , e , e , the dice coefficient loss function is indicated asW , X resembles Tversky loss, Y symbolizes weighted binary cross-entropy, and Z denotes regularization loss.
-Dice coefficient loss: The dice coefficient is utilized to estimate the segmentation performance in the availability of ground truth. The dice coefficient is computed by the expression,
W E, F  	 	(3)

here, the resultant output obtained during the segmentation task is signified as E , and its
corresponding ground truth is denoted by F .

-Tversky loss: It is the generalized form of dice coefficient that helps to attain improved trade- off among recall and precision during the segmentation of large unbalanced OCT image database using CE-Net. The Tversky loss is expressed by,
 
X E, F   	ES  FG	
ES FG   1  ES  FG  1    ES 1  FG 
where, the hyper-parameter is symbolized as  .
 

(4)
 

-Weighted binary cross entropy: It is used to calculate the variation among the actual and predicted binary outcomes. It is computed by,
Y  E, F     h log  FG   1  ES  log 1  FG 	(5) here, h signifies true probability distribution.
-Regularization loss: It is a generic function that helps to increase the generalization performance of CE-Net. Here, the regularisation method called weight decay is used to
 
suppress the weight of CE-Net. The weight decay applies dropout and random transformation to create a large diversified training set.
Thus, the resultant segmented output E obtained is determined from the input OCT image A
S	R
 
using CE-Net and the output extraction.
 
E is further subjected to extract suitable features during feature
 
3.4.	Feature extraction
The redundant or irrelevant features are eliminated from the resultant segmented layer output E by highlighting key patterns and preserving important features via feature extraction. Here, the suitable features are extracted using feature extractors, like reflectivity, thickness, curvature [10], statistical features like mean, correlation, energy, skewness, entropy, and kurtosis [27], and LTXOR [11]. The extraction process executed is described below,
1)	Curvature

The curvature [10] of the segmented layer output E is determined by joining Menger curvature values computed for all points across the layer after smoothening the local weighted polynomial surface and the resultant curvature feature is signified by L1 .
2)	Reflectivity

It is determined from two regions per scan, such as the temporal sides of the foveal peak and the thickest portion of the retina, where the extracted reflectivity feature is denoted as L2 .
3)	Thickness

The Euclidean distance among the corresponding points of the lower and upper boundaries of the segmented layer is computed to determine the thickness of the segmented layer and the extracted thickness feature is given by L3 .
4)	LTXOR

The enhanced version of LXTOR is termed LTXOR [11] feature descriptor, which helps to map the texton shapes more effectively in the Gaussian plane. In LTXOR, the texton images are generated by employing different texton shapes, and the images are then split into overlapping sub-blocks. Then, the position of gray values is updated and the texton picture
 
collects the center of each pixel and its surrounding neighbors after the determination of text on the image. Later, XOR function is applied between the neighbors and center texton. The LTXOR patterns is expressed by,
 
LTXOR
l ,D
 
 l
n1
 
2n1  f S    S  	(6)
T	n	T	z
 

here, the centre and neighbor pixels are signified as  p and z , D designates radius, and the corresponding texton shape is represented as ST  p  and ST z , the XOR function is given by  . Moreover, the texton images are converted into LTXOR maps with values fixed from 0 to 2l 1, here the total neighbors are symbolized as l . Once the LTXOR pattern is computed for pixel i, j  , and the histogram construction is given by,
Q1 Q2
LTXORHist   f LTXOR i, j , k ;	k  0, 2l 1	(7)
i1 j1


where, k signifies random number chosen among 0 and 2l 1, the input size of image is symbolized as Q1  Q2 , and L4 resembles the extracted LTXOR feature.
5)	Statistical features

The statistical features, such as mean, correlation, energy, skewness, entropy, and kurtosis [27] are effectively extracted from the segmented layer and the extraction performed is explicated below,
-Mean: The distribution of concentrated data from the segmented layer is termed mean, which is estimated by,
 
i1
L5  G(d )
d 0
 

(8)
 

here, d signifies the grey scale level of image, G(d ) symbolizes the probability of d , i
signifies the number of grey levels, and the resultant mean feature extracted is denoted as L5 .

-Correlation: The correlation feature is used to express the spatial dependency among the pixels, where L6 denotes the extracted correlation feature and is expressed as,
 
M 1 N 1
 (u, v)P(u, v)  au av
L    u0 v0	
 


(9)
 
6	b b
u v
Here, the normalized value of the gray scale is signified as P u, v, the total pixels along the axis is given by M and N , the standard deviation as well as mean in horizontal spatial domain is symbolized by bu and au , correspondingly the standard deviation as well as mean
for vertical spatial domain is denoted by bv and av .

-Energy: The equality between the image pixel M , N is measured using energy, which is symbolized as L7 and is formulated as,
 

M 1 N 1
L7 	P(u, v)
u 0 v0
 

(10)
 
where, the blocks axis of grayscale is given by u , and v , and P resembles the value of gray level.
-Skewness: The skewness is defined as the distribution of degree of asymmetry of a specific feature around the mean. It is used to compute the symmetry or lack of symmetry in the
 
resultant segmented layer. Thus, the extracted skewness feature is given as as,
 
L8 and is expressed
 
3  i1	3	
 
L8  L5 d  L5 
d 0
 
 G(d )
	(11)
 

-Entropy: The randomness in neighborhood intensity values or textural image is characterized is using the entropy feature, which is given by,
 

M 1 N 1
L9 	P(u, v)  log P(u, v)
u 0 v0


Here, the extracted entropy feature is signified as L9 .
 

(12)
 

-Kurtosis: Kurtosis is used to describe the distribution of the shape of a random variable’s probability and L10 signifies the extracted kurtosis feature and is expressed by,
 
L	 L4  i1 		4 		(13)
 
10	5 
d 0
 
d	L5	G(d )

 

Therefore, the statistical features LF extracted from the segmented layer ES , which is expressed by,

