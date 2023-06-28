# Real-time-2-D-Object-Recognition-cpp
The repository contains the code files for the project aims to develop a computer system that can identify specified objects on a white surface using a downward-facing camera, implementing tasks such as thresholding, cleaning, segmentation, feature computation, training data collection, classification, evaluation, and system demonstration.

## Implementation
The implemented system uses a phone camera to get image frames in real-time and computes the rotation, scale and translation invariant features of the object after thresholding to differentiate the foreground object of interest from the background, essentially giving us a cleaned up segmented binary image of the object. The given implementation has an explicitly defined function for the 2 pass algorithm for segmentation whereas the rest of the implementation uses OpenCV library function. The user is prompted to enter inputs to go into a Training or Classification mode where it can either compute features for new object labels and add to the .csv database file or it can classify the objects in real-time. This is done in a recurring while loop until the project file is stopped. There is also a K-NN implementation of the distance matching as compared to the initial scaled Euclidean distance to calculate the top match for object for classisification. Further, we manually calculate the confusion matrix to examine the performance of the implemented system. In addition to actually objects, additional objects in the database file include paper cut outs of objects to test the system further.


NOTE: The implemented code for the .csv databse file creation ,traversal and manipulation do not include a .h file as the functions are defined in csv_util.cpp inside the same Visual Studio Code project solution. The function prototypes declared at the top of main.cpp call the required functions from csv_util.cpp.
Also note, that the background subtraction implemented assumes the first frame to be the background to be subtracted from the images in the consequent frames that have the object. So it has to be ensured that the initial frame on running the program does not have the object.


## System setup
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/cecdf20a-342e-4230-b02e-7e7da02c83bf)


## Thresholding the input video
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/046f8d0e-4a00-40f7-8384-072e36d2da09)


## Cleaning up the binary image
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/8c221876-9c7c-4fa3-b40f-9bf3dfdf97d2)


## Segmenting the cleaned up image into foregeound object and background
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/3a5d1b70-f740-42d3-9bc2-05d3495b3a44)


##  Compute features 
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/33d32efd-eb51-4e11-9449-d8ae653cb146)


##  Collecting training data
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/880688ee-b9cd-4bb5-b88a-d2f460d27d60)

##  Trained objects saved with labels into csv file with object features(Hu moments followed by mu20_norm, mu02_norm and mu11_norm =>Scale and rotation invariant)
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/4b93d866-2f38-4246-bdbd-8f73a5402e65)


## Classfication mode(Euclidean distance based)
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/115cae98-ae89-47be-b500-bafd6e40acfc)

## KNN based Classification
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/3ccf28a3-2085-4675-824b-f4313c759fc8)

## Performance evaluation with confusion matrices 
The following is the confusion matrix for euclidean distance based classification where the rows of object name are actual inputs to system and the column inputs give the corresponding number of time different unique objects(of the same kind) were labeled by the system
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/4e9e991a-4905-4ef5-9926-2b346da47840)


Confusion matrix of the KNN classifier output is as shown:
![image](https://github.com/josejosepht/Real-time-2-D-Object-Recognition-cpp/assets/97187460/300511be-638e-473a-ad35-aee9789c784a)

## Inference
As we see from analyzing the confusion matrices, the implemented system is not perfect. There are quite a few objects that are being mislabeled. The implemented system also does not take into account for objects it has not be trained for. Even the KNN classifier does not seem to label all the shown objects correctly. But it is a step in the right direction of accuracy and scaled distance for feature matching.
