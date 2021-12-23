![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)  ![seaborn](https://user-images.githubusercontent.com/86251750/132984253-32c04192-989f-4ebd-8c46-8ad1a194a492.png)  ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white) ![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![coursera](https://img.shields.io/badge/Coursera-0056D2?style=for-the-badge&logo=Coursera&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

## Manufacturing Department

* Artificial Intelligence and Machine Learning are transforming the manufacturing industry. According to the report released by World Economic Forum, these technologies will play significant roles in the fourth industrial revolution. Major areas which can be benefited from this are:
      
      Maintenance Department
      Production Department
      Supply Chain Department

* Deep learning has been proven to be superior in detecting and localizing defects using imagery data which could significantly improve the production efficiency in the manufacturing industry.
* Great Example from LandingAI: https://landing.ai/defect-detection/

## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 - [ML by Stanford University ](https://www.coursera.org/learn/machine-learning)

## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Exploring the Data](#viz)
   - [Matplotlib](#matplotlib)
   - [Seaborn](#seaborn)
* [solving the task](#fe)
* [prediction](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

To automate the process of detecting and localizing defects found in Steel manufacturing. Detecting defects would help in improving the quality of manufacturing as well as in reducing the waste due to production defects.

## Dataset Used:<a name="data"></a>

The team has collected images of steel surfaces and thus we can develop a model that could detect and localize defects in real-time and also we have been provided with 12600 images that contain 4 types of defects, along with their location in the steel surface.

## Exploring the Data:<a name="viz"></a>

I have used matplotlib and seaborn visualization skills.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

    pip: pip install matplotlib

    anaconda: conda install matplotlib
    
    import matplotlib.pyplot as plt

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

**Seaborn:**<a name="seaborn"></a>
------
Seaborn is built on top of Python’s core visualization library Matplotlib. Seaborn comes with some very important features that make it easy to use. Some of these features are:

**Visualizing univariate and bivariate data.**

**Fitting and visualizing linear regression models.**

**Plotting statistical time series data.**

**Seaborn works well with NumPy and Pandas data structures**

**Built-in themes for styling Matplotlib graphics**

**The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.**

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install seaborn:

    pip: pip install seaborn

    anaconda: conda install seaborn
    
    import seaborn as sns
    
for more information you can refer to [seaborn](https://seaborn.pydata.org/) official site.

**Screenshots from notebook**

![download](https://user-images.githubusercontent.com/86251750/146676244-d517bf5a-e4c8-4f62-b642-838afdd500fe.png)

![download](https://user-images.githubusercontent.com/86251750/146676269-303d102c-ad24-4d2a-9af6-73da4637712f.png)

## approach for making prediction<a name="fe"></a>
-------

* My first step was to explore the data and gain insights from it.
* BUILDING AND TRAINING A DEEP LEARNING MODEL TO DETECT WHETHER A DEFECT IS PRESENT IN AN IMAGE OR NOT
* ASSESS TRAINED MODEL PERFORMANCE
* BUILDING A RESUNET SEGMENTATION MODEL
* ASSESS TRAINED SEGMENTATION MODEL PERFORMANCE

## Prediction:<a name="models"></a>
------
**LAYERED DEEP LEARNING PIPELINE TO PERFORM CLASSIFICATION & SEGMENTATION**

![image](https://user-images.githubusercontent.com/86251750/146676386-1487d059-4cf3-462f-bc7b-4aaef4c278c8.png)

**CONVOLUTIONAL NEURAL NETWORKS**

* The first CNN layers are used to extract high level general features. 
* The last couple of layers are used to perform classification (on a specific task).
* Local respective fields scan the image first searching for simple shapes such as edges/lines 
* These edges are then picked up by the subsequent layer to form more complex features.

![image](https://user-images.githubusercontent.com/86251750/146676506-4d8c2ee5-12a5-4916-89c9-47f08db56184.png)

**RESNET (RESIDUAL NETWORK)**

* As CNNs grow deeper, vanishing gradient tend to occur which negatively impact network performance.
* Vanishing gradient problem occurs when the gradient is back-propagated to earlier layers which results in a very small gradient. 
* Residual Neural Network includes “skip connection” feature which enables training of 152 layers without vanishing gradient issues. 
* Resnet works by adding “identity mappings” on top of the CNN. 
* ImageNet contains 11 million images and 11,000 categories. 
* ImageNet is used to train ResNet deep network.

![image](https://user-images.githubusercontent.com/86251750/146676549-81329007-a152-4a4b-b05e-444de813bbed.png)

**TRANSFER LEARNING**

* Transfer learning is a machine learning technique in which a network that has been trained to perform a specific task is being reused (repurposed) as a starting point for another similar task.
* Transfer learning is widely used since starting from a pre-trained models can dramatically reduce the computational time required if training is performed from scratch. 

![image](https://user-images.githubusercontent.com/86251750/146648910-1600aa56-abd9-4705-90b0-a7804bec3173.png)

photo credit : [link1](https://commons.wikimedia.org/wiki/File:Lillehammer_2016_-_Figure_Skating_Men_Short_Program_-_Camden_Pulkinen_2.jpg), [link2](https://commons.wikimedia.org/wiki/Alpine_skiing#/media/File:Andrej_%C5%A0porn_at_the_2010_Winter_Olympic_downhill.jpg)

* “Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned”—Transfer Learning, Handbook of Research on Machine Learning Applications, 2009.
* In transfer learning, a base (reference) Artificial Neural Network on a base dataset and function is being trained. Then, this trained network weights are then repurposed in a second ANN to be trained on a new dataset and function. 
* Transfer learning works great if the features are general, such that trained weights can effectively repurposed.
* Intelligence is being transferred from the base network to the newly target network.

*TransferLearning process*

![image](https://user-images.githubusercontent.com/86251750/146647886-2d073768-2de5-4f6e-a086-2b3091903ca0.png)

*Why do we keep the First layer?*

* The first CNN layers are used to extract high level general features. 
* The last couple of layers are used to perform classification (on a specific task).
* So we copy the first trained layers (base model) and then we add a new custom layers in the output to perform classification on a specific new task.

![image](https://user-images.githubusercontent.com/86251750/146647947-315cae13-60d3-48ee-bf43-d06b6ade0660.png)

*TRANSFER LEARNING TRAINING STRATEGIES*

    - Strategy #1 Steps: 
         Freeze the trained CNN network weights from the first layers. 
         Only train the newly added dense layers (with randomly initialized weights).
    - Strategy #2 Steps: 
         Initialize the CNN network with the pre-trained weights 
         Retrain the entire CNN network while setting the learning rate to be very small, this is critical to ensure that you do not aggressively change the trained weights.

Transfer learning advantages are:
- Provides fast training progress, you don’t have to start from scratch using randomly initialized weights
- You can use small training dataset to achieve incredible results

**WHAT IS IMAGE SEGMENTATION?**

* The goal of image segmentation is to understand and extract information from images at the pixel-level. 
* Image Segmentation can be used for object recognition and localization which offers tremendous value in many applications such as medical imaging and self-driving cars etc.
* The goal of image segmentation is to train a neural network to produce pixel-wise mask of the image.
* Modern image segmentation techniques are based on deep learning approach which makes use of common architectures such as CNN, FCNs (Fully Convolution Networks) and Deep Encoders-Decoders.
* we will be using ResUNet architecture to solve the current task. 

![image](https://user-images.githubusercontent.com/86251750/146676783-2956e28e-60d1-4a70-9fae-38a8928e188a.png)

* In CNN for image classification We had to convert the image into a vector and possibly add a classification head at the end. 
* However, in case of Unet, we convert (encode) the image into a vector followed by up sampling (decode) it back again into an image. 
* In case of Unet, the input and output have the same size so the size of the image is preserved. 
* For classical CNNs: they are generally used when the entire image is needed to be classified as a class label. 
* For Unet: pixel level classification is performed.
* U-net formulates a loss function for every pixel in the input image.
* Softmax function is applied to every pixel which makes the segmentation problem works as a classification problem where classification is performed on every pixel of the image. 

Great article by Aditi Mittal: https://towardsdatascience.com/introduction-to-u-net-and-res-net-for-image-segmentation-9afcb432ee2f

**RESUNET**

*ResUNet architecture combines UNet backbone architecture with residual blocks to overcome the vanishing gradients problems present in deep architectures.
*Unet architecture is based on Fully Convolutional Networks and modified in a way that it performs well on segmentation tasks.
*Resunet consists of three parts:
      
    (1) Encoder or contracting path
    (2) Bottleneck 
    (3) Decoder or expansive path 

![image](https://user-images.githubusercontent.com/86251750/146676912-45ab0dc1-d680-4283-bcc3-5a1c434c85ad.png)

![image](https://user-images.githubusercontent.com/86251750/146676940-b8f57bb9-a3bc-4541-b09b-62f1fda102f3.png)

**RESUNET ARCHITECTURE**

1. Encoder or contracting path consist of 4 blocks: 
  
        First block consists of 3x3 convolution layer +  Relu + Batch-Normalization
        Remaining three blocks consist of  Res-blocks followed by Max-pooling 2x2.

2. Bottleneck:
  
        It is in-between the contracting and expanding path.  
        It consist of Res-block followed by up sampling conv layer 2x2.

3. Expanding or Decoder path consist of 4 blocks:

        3 blocks following bottleneck consist of Res-blocks followed by up-sampling conv layer 2 x 2
        Final block consist of Res-block followed by 1x1 conv layer.
        
![image](https://user-images.githubusercontent.com/86251750/146677087-23532b77-bdc3-4425-a713-a70fcf5afd73.png)

RESUNET additional information [link1](https://arxiv.org/abs/1505.04597), [link2](https://arxiv.org/abs/1904.00592), [link3](https://towardsdatascience.com/introduction-to-u-net-and-res-net-for-image-segmentation-9afcb432ee2f)

**MASK**

* The goal of image segmentation is to understand the image at the pixel level. It associates each pixel with a certain class. The output produce by image segmentation model is called a “mask” of the image.
* Masks can be represented by associating pixel values with their coordinates. For example if we have a black image of shape (2,2), this can be represented as: 

![image](https://user-images.githubusercontent.com/86251750/146677999-2738bea4-8d3c-45b3-a194-238f34b5ed19.png)  [[0,0
                                                                                                                   0,0]]

If our output mask is as follows:

![image](https://user-images.githubusercontent.com/86251750/146678527-9479a9b1-5cba-4a98-88df-5155d6b9ed1e.png)  [[255,0
                                                                                                                   0,255]]
                                                                                                                   
* To represent this mask we have to first flatten the image into a 1-D array. This would result in something like [255,0,0,255] for mask. Then, we can use the index to create the mask. Finally we would have something like [1,0,0,1] as our mask.

**RUN LENGHT ENCODING**

* Sometimes it is hard to represent mask using index as it would make the length of mask equal to product of height and width of the image
* To overcome this we use lossless data compression technique called Run-length encoding (RLE), which stores sequences that  contain many consecutive data elements as a single data value followed by the count.
* For example, assume we have an image (single row) containing plain black text on a solid white background. B represents black pixel and W represents white:

                                WWWWWWWWWWWWBWWWWWWWWW
                                WWWBBBWWWWWWWWWWWWWWWWW
                                WWWWWWWBWWWWWWWWWWWWWW
Run-length encoding (RLE):

                                  12W1B12W3B24W1B14W

* This can be interpreted as a sequence of twelve Ws, one B, twelve Ws, three Bs, etc.,

*ASSESS TRAINED MODEL PERFORMANCE*

                precision    recall  f1-score   support

           0       1.00      0.72      0.84       880
           1       0.81      1.00      0.90      1056

    accuracy                           0.87      1936

![download](https://user-images.githubusercontent.com/86251750/146678376-646b28f1-390d-4802-952b-f93876a1ebc8.png)

*Let's show the images along with their original (ground truth) masks*

![download](https://user-images.githubusercontent.com/86251750/146678423-79f39fb8-035d-485c-8ac8-4a0baa3bd2b6.png)
![download](https://user-images.githubusercontent.com/86251750/146678427-b846b82e-f472-483b-bda4-e154e34f698f.png)
![download](https://user-images.githubusercontent.com/86251750/146678431-fe18b3a3-b3d4-4380-9b2f-e6cd2f788575.png)
![download](https://user-images.githubusercontent.com/86251750/146678436-ef72b131-812d-44c9-9e7a-dd36d619ae34.png)
![download](https://user-images.githubusercontent.com/86251750/146678442-cc0107c8-0557-4f76-8d34-a151bab33195.png)
![download](https://user-images.githubusercontent.com/86251750/146678451-b9f6cbcb-fb55-4ed0-b12f-f85cf7a8225e.png)
![download](https://user-images.githubusercontent.com/86251750/146678455-39cd49a7-2ec0-4672-b907-965893954945.png)
![download](https://user-images.githubusercontent.com/86251750/146678459-fe42c8b7-d997-4b17-8c3d-6af1e064b0ef.png)
![download](https://user-images.githubusercontent.com/86251750/146678466-4144fe49-e02f-4d6b-8840-83da946443a4.png)
![download](https://user-images.githubusercontent.com/86251750/146678471-48ee127a-d93b-4d31-bc1b-f228266db094.png)

*visualize the results (model predictions)*

![download](https://user-images.githubusercontent.com/86251750/146678595-2aec124f-1018-421a-9318-4c2603a98f6c.png)
![download](https://user-images.githubusercontent.com/86251750/146678596-e62e8ed7-5a2b-41ea-8e24-5c5a3b3ecaa9.png)
![download](https://user-images.githubusercontent.com/86251750/146678602-77c2344b-6e4f-4523-a37f-44344729922c.png)
![download](https://user-images.githubusercontent.com/86251750/146678607-aed7587a-2b37-424f-9339-165113965387.png)
![download](https://user-images.githubusercontent.com/86251750/146678621-e6a0fcaf-0acf-448a-ba54-23c50ff99a85.png)
![download](https://user-images.githubusercontent.com/86251750/146678630-0012a208-26c7-40e6-8ce3-af39404ab7ee.png)
![download](https://user-images.githubusercontent.com/86251750/146678635-ef714e51-68f4-403a-82ae-b23b90619f76.png)
![download](https://user-images.githubusercontent.com/86251750/146678648-9766814f-b57b-4154-bbd5-bc38e8250f11.png)
![download](https://user-images.githubusercontent.com/86251750/146678653-b46b520c-7111-428d-8b6c-6a3813502559.png)
![download](https://user-images.githubusercontent.com/86251750/146678658-6f125d8d-b845-405e-8027-294abbcd8260.png)


## CONCLUSION:<a name="conclusion"></a>
-----
* we got an accuracy of 87% with a good precision, recall and f1 score after that I build RESUNET SEGMENTATION MODEL which even provide me better result then I save the best model with lower validation loss and used it for prediction.

