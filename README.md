# **Behavioral Cloning** 

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/placeholder.jpg "Model Visualization"
[image2]: ./report_images/center_track_1.jpg "Center lane driving"
[image3]: ./report_images/recovery_1.jpg "Recovery Image"
[image4]: ./report_images/recovery_2.jpg "Recovery Image"
[image5]: ./report_images/recovery_3.jpg "Recovery Image"
[image6]: ./report_images/before_flipping.jpg "Normal Image"
[image7]: ./report_images/after_flipping.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is revised based on the Nvidia network. The revised model consists of 4 convolutional layers and four fully connected layers. (model.py lines 67-83) 

The model includes RELU layers to introduce nonlinearity (code line 72 - 77), and the data is normalized in the model using a Keras lambda layer (code line 70). Then the images are cropped using a Keras cropping2D layer (code line 72). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80). 

One convolutional layer is removed from the original Nvidia net to reduce overfitting as well.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 85-86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, reversed direction driving. Training on the 2nd track was used initially, but discarded at the end because it can add some difficulties when further generating the appropriate amount of data for the vehicle turning at tricky places such as the bridge.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a simple neural network and gradually modify and swith to more complicated neural networks based on the driving performance of fixed data sets. Then adding/removing convolutional layers and dropout layers is performed based on whether the network is overfitted or underfitted.  

My first step was to use a convolution neural network model similar to the Lenet. I thought this model might be appropriate because it has a reasonable performance as the German traffic sign classifier. However, given two rounds of driving in the first track, the vehicle failed to drive in a stable manner in the autonomous mode. Then I switched to Nvidia neural network and the vehcile was at least able to drive straight stably. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has one fewer convolutional layer and one more dropout layer right after the first fully connected layer.

Then I gathered more data by driving in the reverse direction, and another two laps in the default direction.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as at the bridge and the first turning spot beyond the bridge. To improve the driving behavior in these cases, I performed more driving specifically at the places which required sharp turns. This is very necessary because most parts of track 1 is straight which resulted in much more data of the vehicle driving straight. This will
bias the network to drive the vehicle straight more often than it should. Having more data of the vehicle turning will mitigate this bias. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160 x 320 x 3 RGB image   					| 
| Cropping2D            | outputs 65 x 320 x 3 RGB image                | 
| Convolution 5x5     	| 2x2 submsample, valid padding                 |
| Activation    		| relu  										|
| Convolution 5x5	    | 2x2 subsample, valid padding               	|
| Activation    		| relu        						        	|
| Convolution 5x5     	| 2x2 subsample, valid padding               	|
| Activation    		| relu  										|
| Convolution 3x3      	| valid padding                              	|
| Activation    		| relu  										|
| Flatten				| output 400									|
| Fully connected       | output 100       				   	    		|
| Dropout               | 0.5 keepprob                                  |
| Fully connected       | output 50                                     |
| Fully connected       | output 10                                     |
| Fully connected       | output 1                                      |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving captured by the center camera:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the lane once it deviates. These images show what a recovery looks like starting from the 1st to 3rd:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by adjusting the steering measurements for the side cameras' images, normalizing the images to fit the value of each pixel into the range of [-1, 1], and cropping the upper and lower parts of images by 70 and 25 rows of pixels respectively.   


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as the decrease rate on both the training accuracy and validation accuracy became very small. In addition, the trained network was able to guide the vehicle to drive several laps autonomously. I used an adam optimizer so that manually training the learning rate wasn't necessary.
