Complex & Intelligent Systems (2026) 12:65 https://doi.org/10.1007/s40747-025-02188-x 

**~~ORIGINAL ARTICLE~~** 



# **A real-time mobile solution for shoe try-on using foot pose estimation and 3D processing techniques** 

#### **Nguyen Hoang Vu**<sup>**1**</sup> **· Tran Van Duc**<sup>**1**</sup> **· Pham Quang Tien**<sup>**1**</sup> **· Nguyen Thi Ngoc Anh**<sup>**2**</sup> **· Nguyen Tien Dat**<sup>**1**</sup> 

Received: 2 January 2024 / Accepted: 18 November 2025 / Published online: 5 December 2025 © The Author(s) 2025 

#### **Abstract** 

Implementing Augmented Reality (AR) in virtual try-on technology has revolutionized the online shopping experience, transforming how clients engage with products. This technology allows customers to try on clothes without direct physical contact, which has become convenient and valuable in the age of online shopping. Despite clothing being a dominant sector, shoes also hold a significant portion of the market. However, there has been limited previous research conducted on the subject of virtual shoe try-on. This paper presents an innovative solution to the Foot Pose Estimation problem, offering a deep learning model that delivers accurate results while operating in real-time on the CPU. Due to the insufficiency of public foot keypoint datasets, a medium-scale self-collected 2D foot keypoints dataset has been conducted with 9 keypoints each foot for training and evaluating the model. In addition, this model has successfully been utilized to create a shoe AR try-on application for smartphones. It provides a solution that generates a realistic 3D shoe model for a smooth and stable try-on experience. Practical tests have proven that the system gives real-time performance under mobile computing conditions. 

**Keywords** Foot pose estimation · Real-time keypoint detection · Virtual try-on 

## **Introduction** 

COVID-19 pandemic has greatly contributed to the substantial increase in online shopping, leading to a significant shift in consumer behavior in recent times. One important part of online shopping is that customers want to see themselves wearing clothes and accessories. This creates a connection between the virtual world and the real world. In response to this demand, the development of virtual try-on systems has gained substantial momentum. These systems offer a dynamic platform for customers to simulate the act of trying on various clothing and accessory items. This not only offers a novel and efficient shopping experience but also contributes to reducing the rate of returns, a vexing concern for online retailers. While substantial research has been directed 

- B Nguyen Tien Dat datnt65@viettel.com.vn 

Nguyen Thi Ngoc Anh anhnguyenngoc@vnu.edu.vn 

- 1 Viettel High Technology Industries Corporation, Hanoi 100000, Vietnam 

- 2 VNU University of Engineering and Technology, Hanoi 100000, Vietnam 

towards image-based virtual try-on (VTON) techniques [1– 6], the focus predominantly remains centered on clothing, with only a handful of studies exploring the virtual try-on of footwear, such as PITONS [7], which seeks to replicate shoe samples onto photographs featuring the wearer’s feet, using the Generative model as the foundation. However, the computational resource-intensive nature of such models presents a notable barrier to accessibility. 

In the field of the shoe market, shoe AR try-On technology has become an emerging solution that addresses users’ online shoe-shopping needs. This technology allows shoppers to try on shoes without physical impact—which is extremely convenient and useful in the online era. This requires finding the right position and orientation of the user’s foot in front of the camera to render the 3D shoe model, which includes the camera’s rotation and translation matrix. In recent years, some world-wide brands such as Adidas, Nike, and Dior have been investing in shoe AR try-on technology to apply into their websites or mobile applications. Beside that, a start-up company called Wanna,<sup>1</sup> which is founded in 2018, introduced their implementation of 3D and AR technologies in shoe AR try-on with a demo website and soon spread 

> 1 https://wanna.fashion/. 

123 

**65** Page 2 of 22 

Complex & Intelligent Systems (2026) 12 :65 

on multiple platforms including iOS, Android, and WeChat mini-program. Industrial researches on shoe AR try-on technology conducted by world-wide brands and start-ups prove the important this technology in the shoe market. 

Traditionally, foot pose estimation has been achieved through a marker-based motion capture system, that uses sensors to measure and capture the information related to foot movement. Duong et al. [4] proposed a methodology that combines low-cost distance sensors and an inertial sensor unit to estimate foot pose. This approach requires special hardware as well as sensor settings which is also difficult to apply. Furthermore, since most users shop online through mobile devices, developing applications on mobile platforms with real-time processing capabilities is needed for the user’s experience. 

In the past few years, deep learning came up as a general solution for this foot pose estimation problem. In the recent publication [8], the authors present a method for estimating foot pose based on the correlation between 2D and 3D foot keypoints. This approach uses a convolutional neural network (CNN) model capable of extracting precisely 2D-foot keypoint in real-time before using the Perspective-n-Point (PnP) algorithm applied to estimate foot pose. The biggest obstacletothisresearchisthelackofasuitablequalitydataset of 2D foot keypoint, which is not referenced by the author. The available public dataset that includes human foot keypoints, extracted from the original COCO [9] dataset, falls short of adequately addressing the problem. The dataset is not large enough and the quality of the foot keypoints is not good enough to train a deep learning model. Therefore, the authors had to use a 3D Render Tool to create a synthetic dataset of 2D foot keypoints. This dataset is used to train a deep learning model that can extract 2D foot keypoints in real-time. The model is then evaluated on a self-built 2D foot keypoint dataset. The results show that the model can extract 2D foot keypoints in real-time with high accuracy. 

To make the virtual try-on realistic, it’s not only about getting the foot position and direction right but also how the shoes cover the foot naturally. Moreover, the existence of jitter in a real-time system can affect the user’s experience, therefore, a stabilization method is needed to handle this problem. In this work, a deep learning model is proposed with the capability of extracting 2D foot keypoints in realtime on CPU which was trained and evaluated on a self-built 2D foot keypoint dataset. Also to achieve a realistic occlusion, Ray-Casting algorithm is utilized for the virtual camera and 3D shoe model to find the shoe area obscured by the 3D foot model. After that, an alpha-beta filter is applied with a custom Intersection Over Union (IOU) threshold to achieve a simple and efficient stabilization effect that can smooth and eliminate jitters. However, due to the the high product cost of 3D shoe models, our proposed method are applied on a 

limited number of 3D shoe models, which is able to affect the qualitative results in experiments. 

In summary, this research provides three main contributions: (1) a self-built 2D foot keypoints dataset based on a 3D Render Tool Support for labeling and checking data annotation process; (2) A 2D keypoints detection model that works accuratelyinreal-time;(3)Theformulationofamethodology to emulate realistic occlusion in 3D shoe models combining with a mechanism that stabilizes a real-time shoe try-on system, which effectively fixes issues caused by jitter. Additionally, an implementation of a real-time Shoe AR Try-on application on mobile devices improving the user experience when shopping online, which has user interface (UI) illustrated on Fig. 1. This research is divided into five main parts: section “Related work” summarizes previous works on shoe virtual try-on; our self-built foot keypoint dataset is introduced in section “Self-built dataset”; the proposed shoe virtual try-on system is provided in section “Shoe virtual try-on system”; section “Experiment” explains our experimental settings and evaluates our results in both quantitative and quanlitative outcomes; section“Discussions and future work” concludes the paper and opens future works. 

## **Related work** 

### **Keypoints dataset** 

Over the past ten years, many human keypoint datasets have been publicly available and introduced. One of the typical datasets is UIUC [10] data with 539 images (346 for training, 247 for testing), which describe personal data under certain activities such as walking, standing, and the main activity playing badminton. These are very diverse poses, but the activities number is quite limited. Sport Image [11] sports data is more abundant with the addition of sports activities such as football, golf, horseback riding,..., etc. The total number of images in this dataset is 1299 images (649 training, 650 experiments). Some larger datasets like MPII Human Pose [12] include 24,589 images having more than 28,000 labeled people in the training set. MS-COCO dataset [9] 2017 includes 118,287 training images and 5000 images in the evaluation set, totaling more than 150,000 people with approximately 1.7 million keypoints. This dataset provides multiple points on the body part consisting of three keypoints per foot. Despite considerable human pose datasets, the number of foot keypoints is limited and difficult to find in the publicly available data. 

123 



<!-- Start of picture text -->
STUN a © (7ZAAZ____ZLLZ-A-NNN(\<br>cae'AE \ — (a<br>f :¥a<br>v }: Z©<br>ro'} oY<br>}<br>& é9 | v es<br><!-- End of picture text -->

**65** Page 4 of 22 

Complex & Intelligent Systems (2026) 12 :65 

pose. Another approach, known as DPOD [18], estimates the 2D-3D correspondences between an image and available 3D models in a multi-class setting. It utilizes PnP and random sample consensus (RANSAC) algorithms to compute these correspondences and incorporates a custom deep learning-based refinement scheme to further enhance the initial pose estimates. Generally, the above approaches utilize CNN models to estimate the 6DoF of an object based on the relationships between the coordinates of 2D and 3D points. This method achieves high accuracy and generalizability because it is trained on a large amount of data. However, the use of large deep-learning models can impact the real-time performance of the system. In this paper, to address time constraints, the PnP algorithm has been implemented for rapid estimation of the 6DoF of the foot pose relative to the camera. Among many versions of the PnP algorithm given with different advantages and disadvantages, Iterative PnP, also called IPnP [19], using Levenberg Marquardt nonlinear minimization with complexity O(n<sup>5</sup> ) and EPnP [20] algorithm developed with high precision and O(n) linear complexity are two chosen and optimized algorithm in this paper. The PnP algorithm will achieve the best results when the 2D points found a match with the corresponding 3D points. 

### **Deep learning architecture** 

#### **MobileNetV2** 

MobileNet [21–23] is a classification model that can be applied to devices with limited computing power. The first version of MobileNet was released in 2017 [21] popularly known as an efficient and not very computationally intensive convolutional neural network for mobile vision applications. The architecture has a lower number of parameters by millions compared to other neural networks at that time. However, it still maintains good accuracy, which is based on the one called Depthwise Separable Convolution. MobileNetV2 [22] builds upon the foundation of Depthwise Separable Convolutions and introduces additional components such as Linear bottlenecks and Inverted Residual Blocks. This combination illustrates a superior performance and efficiency of MobileNetV2 compared to MobileNet. 

tional Neural Network [24]. This research aims at restoring a high-resolution image with rich details from a single lowresolution image. Pixel shuffle uses deep learning algorithms to extract important information from nearby pixels and rearrange them in a specific order to create a higher-resolution representation. This technique has proven to be successful in preserving and enhancing the image features while upscaling, leading to improved visual fidelity and increased spatial resolution. It is an operation that arranges elements in a shape _(_ ∗ _, C_ × _r_<sup>2</sup> _, H , W )_ to a shape of _(_ ∗ _, C, H_ × _r , W_ × _r )_ . Pixel shuffle has gained popularity in various computer vision tasks, including super-resolution, image generation, and style transfer. Based on the structure and dependencies among adjacent pixels, pixel shuffle can generate highquality images with enhanced details and improved visual fidelity. 

#### **Squeeze and excitation block** 

The Squeeze and Excitation (SE) block was initially introduced in 2017 by Jie Hu et al. in their work on Squeezeand-Excitation Networks (SENets) [25]. This block offers a cost-effective solution for enhancing channel interdependencies in CNNs. Its effectiveness lies in the fusion of spatial and channel information during feature extraction from images. SENets introduce a content-aware mechanism that assigns weights to individual channels, unlike a regular convolution operation that treats each channel equally. Basically, this can involve assigning a single parameter to each channel and assigning a linear scalar value to represent its relevance. After globally compressing each channel, the feature maps are reduced to a single numerical value. This yields a vector of size n, where n corresponds to the number of convolutional channels. The input is then passed through a two-layer neural network, resulting in the generation of a vector of equal dimensions. Now, these n values can serve as weights for the original feature maps, allowing us to scale each channel according to its significance. The SE block presents a simple and efficient addition to any model, reducing the minimal computational burden of the process. Its integration has the potential to enhance the performance of previously built models and enable improved results by retraining pre-built models. 

#### **Pixel shuffle layer** 

### **Virtual try-on techniques** 

Pixel shuffle is a computer vision and image processing technique used to enhance the resolution of low-resolution images. Increasing the size of an image in both width and height can cause a loss of image quality, resulting in a soft and blurry output. Fortunately, with the advancements in deep learning, there are several effective solutions available to address this issue. The pixel shuffle technique was first introduced by Shi et al. in Efficient Sub-Pixel Convolu- 

The virtual try-on community has gained a lot of interest and positive results in recent years in the development of shoe and clothing try-on products. The approach of this technology is divided into two main categories, image-based and video-based. With image-based, a vast of research has been done, for example, VITON [4] and CP-VTON [6] with systems consisting of two blocks—clothing wrapping and try-on 

123 

Page 5 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 

image synthesis, or VTNFP [13], SieveNet [26] has developed a third block—human segmentation generation for the target clothing. Regarding video-based, FW-GAN [27] synthesizes virtual clothes-on videos based on the images and poses of the target person. Besides clothes, different virtual try-on objective has also caught the attention in the last few years such as nail try-on [28]—develops a semantic segmentation of small objects that can run real-time in web mobile application, eyeglasses try-on [29]—enables display 3D glasses by tracking face and head motion of the user, or shoe try-on [8]—uses deep learning model to extract 2D foot keypoints and segment foot area to visualize occluded shoe object. Virtual try-on for fashion accessories such as glasses or shoes have not received as much attention as clothing in general. Furthermore, while the previously mentioned methods may yield favorable outcomes in certain regards, their practical implementation on mobile devices is hindered by the substantial computational expenses involved. A real-time system running on devices with limited resources is proposed in this paper. 

## **Self-built dataset** 

To have the deep-learning model with the capability to extract 2D foot keypoints, training it with a foot keypoint dataset becomes essential. As mentioned in section“Introduction”, the number of public shoe keypoint datasets is limited, a self-build shoe keypoint dataset is essential for our proposed method, which is one of our main contributions. This section will describe the process of building the self-built dataset, including data collection, data annotation, and data augmentation. 

### **Data collection** 

The dataset was created to develop foot keypoint detection models. Due to the constraint of time and devices, various iOS and Android mobile devices are used to collect data videos. To ensure diversity, videos were collected from 22 participants with different ages, genders, costumes, backgrounds, and common foot postures. The image contribution of each category of the self-built dataset is listed on Table 1. These videos were shot from both first and third-person perspectives and have an average length of six seconds, with a ratio of 7:3 for the first view and third view. The dataset currently includes barefoot and socks only, not shoes, sandals, or other accessories. The video will be divided into frames with a gap of five frames. These split frames will then be combined, filtered, and cleaned to remove any blurriness or noise caused by motion. In summary, more than 98 short videos have been collected with 6655 images extracted. The example data image is shown in Fig. 2. 

**Table 1** Image contribution of each category of the self-built dataset 

|Category||No. of images|
|---|---|---|
|Foot|Left|1478|
||Right|1621|
||Both|3556|
|Point of view|First view|4646|
||Third view|2009|
|Gender|Male|3793|
||Female|2882|
|Cloth type|Long|3783|
||Short|2882|
|Background|Indoor|5344|
||Outdoor|1311|



### **Data annotation** 

The keypoint data is structured similarly to the COCO dataset [9], comprising 18 key points representing 9 points per leg with 18 connections, as depicted in Fig. 3. These data points are defined in the context of the foot’s threedimensional space, encompassing areas like the big toe, little toe, instep, heel, and adjacent regions. The order of key points is as follows: big toe, little toe, near little toe side, near big toe side, far big toe side, far little toe side, dorsum, heel, and upper heel. The selection of these key points is guided by their ease of recognition and distinctive characteristics when projected from 3D to 2D space. 

The primary objective behind defining these key points is to establish a 6DoF system for the feet, thereby enhancing the algorithm’s accuracy in addressing the perspective-n-point problem. During the data labeling process, obscured points on the foot may arise due to varying viewing angles. Based on the degree of occlusion and the ability to estimate the location of the occluded point, each point is labeled as visible or not visible, drawing on personal experience. 

To facilitate the labeling process, a custom label tool named "3D Render Tool" (shown in Fig. 4) was developed, involving two main functions: Support for labeling and checking labeled data. For labeling purposes, firstly, the "3D Render Tool" places a 3D standard foot object manually in the correct position and direction so that its projection fit to the input image’s foot area. For convenience, an automatic mapping module is utilized to determine raw position and direction of 3D standard foot. This module firstly predict the 2D foot keypoints through a network trained on small datasets, and then use PnP algorithm to generate position and direction of 3D foot. Rotated and translated 3D foot can be refined via three its axes (top left frame in Fig. 4), so that achieve the desire fitting in the given image (shown in right frame in Fig. 4). Secondly, the defined 3D points 

123 

**65** Page 6 of 22 

Complex & Intelligent Systems (2026) 12 :65 



**Fig. 2** Foot image dataset 

123 



<!-- Start of picture text -->
o \ iSye SSZ4\\ AW\ <AZA<br><!-- End of picture text -->



<!-- Start of picture text -->
P a R e te e es in Pee iS ce: pe a<br><!-- End of picture text -->



<!-- Start of picture text -->
Y<br>a<br>y4<br>+ve = Iso<br>| \\ AY i<br>| Bo eee<br>=e OG ae NEXT APPLY<br><!-- End of picture text -->



<!-- Start of picture text -->
Model Shoe 3D<br>Input ee Output<br>Sd on Dawn cctimetine EE ap oreo einen HO Lan Doeeremee es)<br>_ , 2D Pose Estimation 1: 3D Pose Estimation |, | 3DPost-process oe<br>CEN; | 1 | MobilePoseSingle LocalizationKeypoints | 44W | Estimation6DoF Stabilization3D Pose | y1rnI | Generation3D Osélision I!aa’a yg| i,<br>Fo rrbee ee eee eee! I.E<br><!-- End of picture text -->



<!-- Start of picture text -->
Input<br>=<br>= \\\\\<br>Bri<br>ul<br><!-- End of picture text -->



<!-- Start of picture text -->
Data Input 2D Foot Pose<br>a Ce ee ee —_<br>__, 2D Pose Estimation Module ! oe<br>gat ee i I v © 2 ec<br>a . . I \ ; fin Ss<br>fe<br>5 ! Single Keypoint I 4 N\A<br>MobilePose Localization oo | 4<br><!-- End of picture text -->



<!-- Start of picture text -->
fi<br><!-- End of picture text -->



fod 

**65** Page 10 of 22 

Complex & Intelligent Systems (2026) 12 :65 



**Fig. 9** Foot PAF 

Consider a single connection shown in Fig. 9, let x _j_ 1 and x _j_ 2 be the ground-truth location of foot parts _j_ 1 and _j_ 2 from connection _c_ in the image. The value at **L**<sup>∗</sup> _c_<sup>_(_</sup><sup>**p**</sup><sup>_)_is a unit vector</sup> if point **p** lies on the pair that points from _j_ 1 to _j_ 2, otherwise the value is zero-valued. 

To evaluate _f L_ in Eq.(7) during training, the ground-truth PAF, **L** _c_ at image point **p** will be defined as: 



Here, **v** = _(_ x _j_ 2 − x _j_ 1 _)/_ ∥x _j_ 2 − x _j_ 1 ∥2 is the unit vector in the direction of the connection. The set of points on the pair is defined as those within a distance threshold of the line segment. Those points **p** satisfy the condition as: 



where the connection width _σl_ is a distance in pixels, and the connection length is _lc_ = ∥x _j_ 2 − x _j_ 1 ∥2 and **v** ⊥ is a vector perpendicular to **v** . 

#### **Keypoint localization** 

The Keypoint localization block has the goal of determining the final 2D foot keypoint localization based on the extracted heatmap and pafmap of the previous block. By using the information of vector for each pair of points in pafmap, we can correctly select the correct keypoint position and eliminate noise in heatmap. 

First, non-maximum suppression is performed on predicted heatmaps to find all local maximum peaks. For each channel of heatmaps, the number of candidates may be higher than one due to a misunderstanding of the model. A large number of sets of possible connections are defined based on these candidates. In general, to find valid set of keypoints and connections related to right and left foot, find valid connections [30] and depth first search algorithm are applied with the process as Algorithm 1. 

**Algorithm 1** Find pair of foot function 

**Require:** _paf map_ **Require:** _jointList Per JointT ype_ **Ensure:** _right Foot, lef t Foot_ 

_validConnectionList_ ←{} 

**for** _type_ in _def inedT ypeConnection_ **do** 

_jointsSrc, jointsDst_ ← _jointList Per JointT ype(type)_ **for** _jointSrc_ in _jointsSrc_ **do for** _joint Dst_ in _jointsDst_ **do** Sample the line segment between two points of the connection to find _n_ interpolated points. _lines_ ← _LineSegFunct( jointsSrc, jointsDst) count_ ← 0 _v_ ← ∥ _joint Dstjoint Dst_ −− _jointSrcjointSrc_ ∥2 **for** _line_ in _lines_ **do** Check if the vector of the line from pafmap has the same direction as of the connection 

_v_ 1 ← _paf map(line)_ 

**if** _v_ 1 × _v > paf T hresh_ **then** 

_count_ ← _count_ + 1 **end if end for if** _count > thesh_ **then** _connection_ ← _( jointSrc, joint Dst)_ validConnectionList.insert(connection) **end if end for end for end for** 

_right Foot, lef t Foot_ ← _Foot DFSFunct(validConnectionList)_ ▷ Consider joint and connection as a graph structure, apply DFS to associate connection to the same foot together 

Since all the key points are joined into pairs as Fig. 10 shows, pairs that share the same part detection candidates are assembled into the correct foot. An empty list is created to store the key points for each foot. Go over each pair, and check if part A of the pair is already present in any of the lists. If it is present, then it means that the key point belongs 

123 



<!-- Start of picture text -->
F 7 F 7<br>FO F1<br><!-- End of picture text -->

**65** Page 12 of 22 

Complex & Intelligent Systems (2026) 12 :65 

#### **3D pose stabilization** 

The importance of the 3D Pose Stabilization block becomes apparent when working with a sequence of images in reallife applications, particularly when directly streaming from a camera. This eliminates jitter from the continuous system output,resultinginasmootherandmoreconsistentuserexperience. Jitter is the inconsistency in the time interval between consecutive images in a real-time system, which negatively impacts the performance and the experience of the user. To prevent jitter, the Alpha-Beta filter is used to stabilize and smooth values in R, T based on consecutive input, and output of the system. 

this issue, a motion detection block is employed, relying on the Intersection over Union (IOU) value between consecutive foot area frames in the image sequence. The IOU is calculated through the foot bounding box of the image sequence. In particular, the foot bounding box in the image sequence is compared with an image that has an interval of 10 images to the past. A movement threshold value of 0.96 is established, signifying that the foot is considered stable if the IOU score surpasses this threshold. When the foot is deemed stable, the parameters of both rotation (R) and translation (T) are no longer updated, ensuring stability in the algorithm. 

### **3D pose-process module** 

R, T are defined as: 



The total state variables required to update is 



Here, assume that the velocity of change with respect to different values in R and T is constant, the state of values and velocity of R and T is updated following the State Update Equation for position and velocity respectively. The _α_ − _β_ track update equations are defined as: 

Details of the 3D Pose-process Module are shown in Fig. 11. 3D shoe models, having color textures and manually preprocessed to fit the standard foot object shape, are transformed by applying R, T in the previous module to get the 3D shoe and foot object with the position and direction corresponding to the foot in the image. Then a virtual camera is set up at coordinate origin with fixed intrinsic parameters that used in PnP algorithm. The occlusion utilizes the Ray-Casting algorithm for the virtual camera and 3D shoe model to find the shoe area obscured by the 3D foot model. Finally, the resulting image is obtained by rendering the visible shoe area and then merging it with the original image. 

The Stage Update Equation for position: 



## **Experiment** 

### **Experimental setting** 

The State Update Equation for velocity: 



In (15) and (16), _xn_ , _x_ ˙ _n_ , _zn_ present state, speed change of that state, and system result respectively at iteration n. Additionally, the need for _α_ and _β_ values are to control the error in system measurement. When the system measurement range is not as expected, probably due to two reasons: The imprecision in measurement or the change in velocity of the user’s foot. The factor _β_ in (16) is determined by the radar precision level. If the precision of radar is high the gap between the predicted and measured range may be the result of velocity change. In this case, the _β_ value should be set high. In reverse, the low _β_ is set. The factor _α_ in (15) depends on the radar measurement precision to control the change in the measured range. For high-precision radar, high _α_ should be chosen, giving high weight to the measurements. 

Although the Alpha-beta algorithm effectively reduces most of the foot movement jitter, it still struggles to eliminate noise when the foot remains entirely stationary. To address 

**Dataset** : Due to the limitation of public foot keypoint datasets,onlyourself-builtdatasetdescribedinsection“Selfbuilt dataset” is selected for experiments and evaluation. The data is divided into three sets, 33,274 argumented training images from 5799 original images, 577 evaluation images, and 279 test images. The images in each set do not have overlap characteristics to ensure data independence. In addition, the data in the test and evaluation sets are not augmented to ensure the training goal. All images are padded zero-value and resized to 256x192 resolution. 

**Training** : In order to train the model, AdamW opimization [32] is used with weight decay 1 × 10<sup>−4</sup> , momentum variables _B_ 1 = 0 _._ 9 _B_ 2 = 0 _._ 999. AdamW is an extended version of Adam optimization, which improved the generalization and convergence properties compared to Adam. The learning rate is initialized as 1 × 10<sup>−4</sup> , with an Exponential Decay Learning rate scheduler. The involved experiment was performed on platforms including an NVIDIA 1080 GPU (8GB), an Intel Core i5-8500 CPU (3.00GHz x 6). The network was trained with loss L2 as depicted in sec- 

123 



<!-- Start of picture text -->
Post proce<br>i inpus OP a !<br>‘leer ——“hee<br>| | |! 3DPost-proess.sts=—<‘S; SC*é*é‘;é*s<br>| |<br>| |<br>™ | | ——— a AN |<br>|<br>as { \\ \ | | 3D<br>Ef) \\\ a . 5 Occtusion F |? \\ \ |<br>ey Title My<br>| \ WY<br>| Wea | Wire |<br>| We a | | eee =|<br>| —_—_— - +!<br>.___ |_._ I, oe —---------!<br><!-- End of picture text -->



<!-- Start of picture text -->
Mm Single MobilePose 63.64 65.46<br>60+ mmm@—l SingleSingle MobilePoseMobilePose No2No3 57.79 59.95 59'24-<br>52.35<br>50 : |<br>41.82<br>39.83<br>~& 40 36.38<br>nad ate 31.5 33.19 31632 33.25<br>o 29.98 po :<br>on 30<br>20<br>0<br>CPU only CPU with GPU Iphone 11 Iphone 11 Pro Ipad Pro 11 M1<br>Devices<br><!-- End of picture text -->



<!-- Start of picture text -->
0,95<br>0,85<br>3<br>0,75<br>o<br>x<br>a)<br>&<br>3<br>OO 0,65<br>Ge<br>°<br>svo —®SingleMobilePose 3PAF_1HEAT<br>5<br>2 0,55<br>a<br>—*—SingleMobilePose_1PAF_1HEAT<br>0,45 . :<br>—*SingleMobilePose_2PAF_1HEAT<br>0,35 —®OpenPoseVGG<br>0,25<br>2 3 4 5 6 1 8 9 10 11<br>Threshold (pixels)<br><!-- End of picture text -->



<!-- Start of picture text -->
Rx Ry — R1_Origin<br>0.095 0.05 7 —— R1_Kalman<br>0.090<br>0.00<br>0.085<br>0.080 0.05<br>0.075 -0.10<br>0.070<br>-0.15<br>0.065 7 —— RO Origin<br>0.060 — RO_Kalman -0.20<br>0 50 100 150 200 250 300 350 0 50 100 150 200 250 300 350<br>Frame number Frame number<br>Rz TX |— TO_Origin<br>— TOKalman<br>0.44<br>0.2<br>0.42<br>0.40 0.0<br>0.38<br>-0.2<br>0.36<br>0.34 — R2_Origin -0.4<br>— R2_Kalman<br>0 50 100 150 200 250 300 350 0 50 100 150 200 250 300 350<br>Frame number Frame number<br>Ty Tz<br>2<br>2<br>1<br>i<br>0 —— T1_OriginT1_Kalman a —— T2_OriginT2_Kalman<br>-1<br>-1<br>-2<br>-2<br>-3<br>0) 50 100 150 200 250 300 350 ) 50 100 150 200 250 300 350<br>Frame number Frame number<br><!-- End of picture text -->



<!-- Start of picture text -->
RX | — Roorigin Ry |— zi origin<br>0.095 | —— ROKalmanlou 0.05 7 —— R1_Kalmanlou<br>0.090<br>0.00<br>0.085<br>0.080 —0.05<br>0.075 -0.10<br>0.070<br>-0.15<br>0.065<br>0.060 -0.20<br>0 50 100 150 200 250 300 350 0 50 100 150 200 250 300 350<br>Frame number Frame number<br>Rz |(_— R2_Origin Tx |— TO_Origin<br>— R2_KalmanlOU — TO_KalmanloU<br>0.44<br>0.2<br>0.42<br>0.0<br>0.40<br>-0.2<br>0.38<br>-0.4<br>0.36<br>0 50 100 150 200 250 300 350 ) 50 100 150 200 250 300 350<br>Frame number Frame number<br>Ty Tz<br>2<br>2<br>1<br>s |<br>0 —— T1_Origin1 Kalmaniou a —— T2_Origin12_Kalmanlou<br>=i<br>-1<br>9<br>2<br>-3<br>0 50 100 150 200 250 300 350 0 50 100 150 200 250 300 350<br>Frame number Frame number<br><!-- End of picture text -->



<!-- Start of picture text -->
12<br>— Ours<br>10H émezon<br>— Adidas<br>8<br>£6<br>iS<br>ae|<br>2<br>4<br>2<br>)<br>) 1 2 3 4 5<br>Score<br><!-- End of picture text -->

|Target Shoes|Input|Result|Input|Result<br>Input<br>Result|
|---|---|---|---|---|
|es||A|Pa|le<br>:|
||a A|be|l “|1,<br>\e|
|iae—|VA|e|ei <br>ee|AZ A1S eA<br>AG?<br>Zo<br> leg|<br>4A Ve|
||Kw|«K“\|eax|EBX<br>heGe|





<!-- Start of picture text -->
Adidas e<br>=)— LT Hi ily Ss Y] Wf yWi) Sy Yy YG ypii SyWINN ANUN:<br>if<br>a Sy ad SS sl, NEURG18 :i Ye |<br>a SS<br>SS2x<br>— SN 7 —S<br>SSS SSSSS Color: iris MEEESANS<br>°<br>= |<br>i { R . =Hl Hl : fi Hay<br>a 4<br>| a i<br>\ i<br>10:12<br>S| aS p. Ny 4<br>Le ZZ<br>. Ba Be 4 > -<br>CK NN ZN > \ a oA<br>Ve No“oa ~ NY Ww Zz<br>: ss]<br><!-- End of picture text -->

Page 21 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



**Fig. 19** An example of a failure case of the proposed method 

**Acknowledgements** This research was fully funded by Viettel High Technology Industries Corporation. The authors would like to thank all members of the 3DR team for their contribution. 

### **Declarations** 

**Open Access** This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/. 

## **References** 

1. Ge Y, Song Y, Zhang R, Ge C, Liu W, Luo P (2021) Parser-free virtual try-on via distilling appearance flows. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 8485–8493 

2. Ge C, Song Y, Ge Y, Yang H, Liu W, Luo P (2021) Disentangled cycle consistency for highly-realistic virtual try-on. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 16928–16937 

3. Han X, Hu X, Huang W, Scott MR (2019) Clothflow: a flowbased model for clothed person generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 10471–10480 

4. Han X, Wu Z, Wu Z, Yu R, Davis LS (2018) Viton: an image-based virtual try-on network. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 7543–7552 

5. Hsieh C-W, Chen C-Y, Chou C-L, Shuai H-H, Liu J, Cheng W- H (2019) Fashionon: semantic-guided image-based virtual try-on with detailed human and clothing information. In: Proceedings of the 27th ACM International Conference on Multimedia, pp 275– 283 

6. Wang B, Zheng H, Liang X, Chen Y, Lin L, Yang M (2018) Toward characteristic-preserving image-based virtual try-on network. In: Proceedings of the European Conference on Computer Vision (ECCV), pp 589–604 

7. Chou C-T, Lee C-H, Zhang K, Lee H-C, Hsu WH (2019) Pivtons: Pose invariant virtual try-on shoe with conditional image completion. In: Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part VI 14, Springer, pp 654–668 

8. An S, Che G, Guo J, Zhu H, Ye J, Zhou F, Zhu Z, Wei D, Liu A, Zhang W (2021) Arshoe: real-time augmented reality shoe try-on system on smartphones. In: Proceedings of the 29th ACM International Conference on Multimedia, pp 1111–1119 

9. Lin T-Y, Maire M, Belongie S, Hays J, Perona P, Ramanan D, Dollár P, Zitnick CL (2014) Microsoft coco: common objects in context. In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, Springer, pp 740–755 

10. Tran D, Forsyth D (2010) Improved human parsing with a full relational model. In: Computer Vision–ECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part IV 11, Springer, pp 227– 240 

11. Hidalgo G, Raaj Y, Idrees H, Xiang D, Joo H, Simon T, Sheikh Y (2019) Single-network whole-body pose estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 6982–6991 

12. Andriluka M, Pishchulin L, Gehler P, Schiele B (2014) 2d human pose estimation: new benchmark and state of the art analysis. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 3686–3693 

13. Yu R, Wang X, Xie X (2019) Vtnfp: an image-based virtual try-on network with body and clothing feature preservation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 10511–10520 

14. Xu Y, Zhang J, Zhang Q, Tao D (2022) Vitpose: simple vision transformer baselines for human pose estimation. arXiv preprint arXiv:2204.12484 

15. Sun K, Xiao B, Liu D, Wang J (2019) Deep high-resolution representation learning for human pose estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5693–5703 

16. Rad M, Lepetit V (2017) Bb8: a scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth. In: Proceedings of the IEEE International Conference on Computer Vision, pp 3828–3836 

17. Tekin B, Sinha SN, Fua P (2018) Real-time seamless single shot 6d object pose prediction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 292–301 

18. Zakharov S, Shugurov I, Ilic S (2019) Dpod: 6d pose object detector and refiner. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1941–1950 

19. Madsen K, Nielsen HB, Tingleff O (2004) Methods for non-linear least squares problems 

20. Lepetit V, Moreno-Noguer F, Fua P (2009) Ep n p: an accurate o 

- (n) solution to the p n p problem. Int J Comput Vision 81:155–166 

- 21. Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, Andreetto M, Adam H (2017) Mobilenets: efficient convolutional 

123 

**65** Page 22 of 22 

Complex & Intelligent Systems (2026) 12 :65 

neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 

22. Sandler M, Howard A, Zhu M, Zhmoginov A, Chen L-C (2018) Mobilenetv2: inverted residuals and linear bottlenecks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 4510–4520 

23. Howard A, Sandler M, Chu G, Chen L-C, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V et al (2019) Searching for mobilenetv3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1314–1324 

24. Shi W, Caballero J, Huszár F, Totz J, Aitken AP, Bishop R, Rueckert D, Wang Z (2016) Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 1874–1883 

25. Hu J, Shen L, Sun G (2018) Squeeze-and-excitation networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 7132–7141 

26. Jandial S, Chopra A, Ayush K, Hemani M, Krishnamurthy B, Halwai A (2020) Sievenet: a unified framework for robust image-based virtual try-on.In: Proceedingsofthe IEEE/CVFWinterConference on Applications of Computer Vision, pp 2182–2190 

27. Dong H, Liang X, Shen X, Wu B, Chen B-C, Yin J (2019) Fw-gan: flow-navigated warping gan for video virtual try-on. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1161–1170 

28. Duke B, Ahmed A, Phung E, Kezele I, Aarabi P (2019) Nail polish try-on: realtime semantic segmentation of small objects for native and browser smartphone ar applications. arXiv preprint arXiv:1906.02222 

29. Kobayashi T, Sugiura Y, Saito H, Uema Y (2019) Automatic eyeglasses replacement for a 3d virtual try-on system. In: Proceedings of the 10th Augmented Human International Conference 2019, pp 1–4 

30. Cao Z, Simon T, Wei S-E, Sheikh Y (2017) Realtime multi-person 2d pose estimation using part affinity fields. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 7291–7299 

31. Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L (2009) Imagenet: a large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp 248–255. Ieee 

32. Loshchilov I, Hutter F (2017) Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 

**Publisher’s Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

123 

