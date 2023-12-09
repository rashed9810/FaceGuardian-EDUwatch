Introduction
Attendance being a very necessary side of administration may normally become an arduous, redundant activity, pushing itself to inaccuracies. The traditional approach of making roll calls
proves itself to be a statute of limitations as it is very difficult to call names and maintain its record especially when the ratio of students is high. Every organization has its way of taking
measures for the Attendance of students. Some organizations use document-oriented Approach and others have implemented these digital methods such as biometric fingerprinting techniques and card swapping techniques. However, these methods prove to be a statute of limitations as it subjects students to wait in a time-consuming queue. So, why not shift to an
automated attendance system which works on face recognition technique? Be it a classroom or
entry gates, it will mark the attendance of the students, professors, employees, etc. The main objective of this project is to develop face recognition based automated student attendance system.

METHODOLOGY

Limitations of the Images
The input image for the proposed approach has to be frontal, upright and only a single face. Although the system is designed to be able to recognize the student with glasses and without glasses, student should provide both facial images with and without glasses to be trained to increase the accuracy to be recognized without glasses. The training image and testing image should be captured by using the same device to avoid quality difference. The students have to register in order to be recognized. The enrolment can be done on the spot through the user-friendly interface.
These conditions have to be satisfied to ensure that the proposed approach can perform well.

Face Detection 
Viola-Jones object detection framework will be used to detect the face from the video camera recording frame. The working principle of Viola-Jones algorithm is mentioned in Chapter 2. The limitation of the Viola-Jones framework is that the facial image has to be a frontal upright image, the face of the individual must point towards the camera in a video frame.
In this use-case we will try to detect the face of individuals using the haarcascade_frontalface_default.xml

Pre-Processing
Testing set and training set images are captured using a camera. There are unwanted noise and uneven lighting exists in the images. Therefore, several pre-processing steps are necessary before proceeding to feature extraction.

Pre-processing steps that would be carried out include scaling of image, median filtering, conversion of colour images to grayscale images and adaptive histogram equalization. The details of these steps would be discussed in the later sections..

Scaling of Image
Scaling of images is one of the frequent tasks in image processing. The size of the images has to be carefully manipulated to prevent loss of spatial information. (Gonzalez, R. C., & Woods, 2008), In order to perform face recognition, the size of the image has to be equalized. This has become crucial, especially in the feature extraction process, the test images and training images have to be in the same size and dimension to ensure the precise outcome. Thus, in this proposed approach test images and train images are standardize at size 250 × 250 pixels.

scale Factor — Parameter specifying how much the image size is reduced
at each image scale. Basically, the scale factor is used to create your scale pyramid. More explanation, your model has a fixed size defined during training, which is visible in the XML. This means that this size of the face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm. 1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce the size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether. In our case, I have used 1.0485258 as the scaleFactor as this worked perfectly for the image that I was using.



















Conversion to Grayscale Image

Camera captures color images, however the proposed contrast improvement method CLAHE can only be performed on grayscale images. After improving the contrast, the illumination effect of the images able to be reduced. LBP extracts the grayscale features from the contrast improved images as 8 bit texture descriptor (Ojala, T. et al., 2002).Therefore, color images have to be converted to grayscale images before proceeding to the later steps. By converting color images to grayscale images, the complexity of the computation can be reduced resulting in higher speed of computation (Kanan and Cottrell, 2012). Figure 3.4 shows the conversion of images to grayscale image.


 
	Figure 3.8 Conversion of Image to Grayscale Image source(https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_rgb_to_gray.html)

Generally the images that we see are in the form of RGB channel(Red, Green, Blue). So, when OpenCV reads the RGB image, it usually stores the image in BGR (Blue, Green, Red) channel. For the purposes of image recognition, we need to convert this BGR channel to gray channel. The reason for this is gray channel is easy to process and is computationally less intensive as it contains only 1-channel of black-white.
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
Here the parameters to the function cvtColor will be the image variable name (resized in our case) and the COLOR_BGR2GRAY.

Feature Extraction

Now after converting the image from RGB to Gray, we will now try to locate the exact features in our face.

Different facial images mean there are changes in textural or geometric information.
In order to perform face recognition, these features have to be extracted from the facial
images and classified appropriately. In this project, enhanced LBP is used for face recognition. The idea comes from the nature of human visual perception which performs face recognition depending on the local statistic and global statistical features.
Enhanced LBP extracts the local grayscale features by performing feature extraction on a small region throughout the entire image.


The LBPH uses 4 parameters:
Radius: the radius is used to build the circular local binary pattern and represents the radius around the central pixel. It is usually set to 1.
Neighbors: the number of sample points to build the circular local binary pattern. Keep in mind: the more sample points you include, the higher the computational cost. It is usually set to 8.
Grid X: the number of cells in the horizontal direction. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector. It is usually set to 8.
Grid Y: the number of cells in the vertical direction. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector. It is usually set to 8.

Now,using the image generated in the last step, we can use the Grid X and Grid Y parameters to divide the image into multiple grids, as can be seen in the following image:
 

Based on the image above, we can extract the histogram of each region as follows:
•	As we have an image in grayscale, each histogram (from each grid) will contain only 256 positions (0~255) representing the occurrences of each pixel intensity.
•	Then, we need to concatenate each histogram to create a new and bigger histogram. Supposing we have 8x8 grids, we will have 8x8x256=16.384 positions in the final histogram. The final histogram represents the characteristics of the image original image.

Training the Algorithm: First, we need to train the algorithm. To do so, we need to use a dataset with the facial images of the people we want to recognize. We need to also set an ID (it may be a number or the name of the person) for each image, so the algorithm will use this information to recognize an input image and give you an output. Images of the same person must have the same ID. With the training set already constructed, let’s see the LBPH computational steps.
Training the Algorithm: First, we need to train the algorithm. To do so, we need to use a dataset with the facial images of the people we want to recognize. We need to also set an ID (it may be a number or the name of the person) for each image, so the algorithm will use this information to recognize an input image and give you an output. Images of the same person must have the same ID. With the training set already constructed, let’s see the LBPH computational steps.
Performing the face recognition: In this step, the algorithm is already trained. Each histogram created is used to represent each image from the training dataset. So, given an input image, we perform the steps again for this new image and creates a histogram which represents the image.
•	So to find the image that matches the input image we just need to compare two histograms and return the image with the closest histogram.
•	We can use various approaches to compare the histograms (calculate the distance between two histograms), for example: euclidean distance, chi-square, absolute value, etc. In this example, we can use the Euclidean distance (which is quite known) based on the following formula:

 
•	So the algorithm output is the ID from the image with the closest histogram. The algorithm should also return the calculated distance, which can be used as a ‘confidence’ measurement. Note: don’t be fooled about the ‘confidence’ name, as lower confidences are better because it means the distance between the two histograms is closer.
•	We can then use a threshold and the ‘confidence’ to automatically estimate if the algorithm has correctly recognized the image. We can assume that the algorithm has successfully recognized if the confidence is lower than the threshold defined.
User interface and user experience
In this proposed approach, face recognition student attendance system with user-friendly interface is designed by using PyQt5 framework.
In this project we are going to perform Facial recognition with high accuracy and that will use webcam to detect faces and record the attendance live in an excel sheet. The user will be able to resister himself/herself with their photo and the attendance will be recorded if the face matches using the live camera. This project will have a GUI to make it user friendly.
Modules:
1.Admin Module: The admin will be able to log in into the admin module through the main interface. The admin module will have 2 buttons- 1. Registration 2. Attendance report.  The registration button will bring up an interface by which admin will register students by providing their information and capturing their photos and finally train the photos with the train button for face recognition. The attendance report button will bring up an interface through which the admin will be able to see all the students information along with their attendance and working time and date and print it. 

2. Attendance Module: In this module the student will be able to give their attendance by facing the camera and checking in by using the ‘Check In’ button and can also check out by using ‘Check Out’ button, which will allow to record their working time. It will also show some information related to the student.
 
3. Report Generation Module: This module will have an interface through which user will be able to view attendance and some information a student by using their unique student id number and print them.












Interfaces

 


 























Flow model


Admin Interface:
 
Instructor Interface:
 

Student Interface:
 













Use Case
 


Activity Diagram
 
Swimlane Diagram
 
















Architectural Design
 












Component Design
 

















Deployment Design

 















Conclusion

In this approach, a face recognition based automated student attendance system is thoroughly described. The proposed approach provides a method to identify the individuals by comparing their input image obtained from recording video frame with respect to train image. This proposed approach able to detect and localize face from an input facial image, which is obtained from the recording video frame. Besides, it provides a method in pre-processing stage to enhance the image contrast and reduce the illumination effect.








