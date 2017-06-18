**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[distribution]: ./writeup_plots/distribution.png "Distribution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/johangithub/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic Summary of the Dataset

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (32 px by 32 px with RGB channels)
* The number of unique classes/labels in the data set is 43. See [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/signnames.csv) for the sign names in the dataset

#### 2. Training Label Distribution

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of signs in the dataset. Notice that training dataset are not equally distributed. Signs such as 1,2 (Speed limit) and 38 (Keep right) are more abundant than, say, sign 19 (Dangerous curve to the right). This is important to keep in mind for a couple of reasons: 1) We could potentially overfit on signs that appear more often than the ones that appear less, and 2) The distribution of the training set may not match the distribution of test set or the real world data.   

![alt text][distribution]

### Design and Test a Model Architecture

#### 1. Preprocessing.

I did not do any preprocessing except one crude normalization as suggested:  X_train = X_train - 128 /  128. I did not perform grayscaling of the training dataset as I wanted to conserve the information contained in RGB channels. No augmentation was done, but I do recognize that it is an important step for the actual production system as you want your Computer Vision system to be resilient against rotation, color perturbation, etc. 

#### 2. Model Architecture

My final model was inspired by VGG net.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3       | 1x1 stride, same padding, outputs 32x32x64    |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x128   |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x128   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x128                  |
| Fully connected       | Input: 8x8x128, flattens and outputs 8192     |
| Dropout               | Probability: 0.5                              |
| Fully connected       | Input: 8192, outputs 43                       |
| Dropout               | Probability: 0.5                              |
| Softmax               |                                               |


---DELETE

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


After 100 epochs, my validation set accuracy reached 0.961 with the test set accuracy of 0.937

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
---DELETE

#### 3. Training Approach
To train the model, I used a GTX 1060 I personally own. Training the data with the architecture mentioned above took about 5 seconds per epoch. I trained to 100 epochs which took about 8 minutes to complete. While I did not do any grid search for hyperparameters, I found that learning rate of 5e-4 was suitable, as it seemed more stable than 1e-3 and quicker to learn than 1e-4. Batch size used was 128 as given.

Initially, I chose LeNet as given in the project, but I wanted to try to implement more of a recent model architecture that was simple enough to implement on raw tensorflow from scratch, thus VGG net was chosen. Compared to the actual VGG net, this architecture skips a few steps such as a couple of conv layers in the middle and the last fully connected layer. While training, I saw that the model complexity did not necessarily increase in test accuracy, perhaps due to the traffic sign dataset containing much less information than the Imagenet dataset (32x32x3 vs 224x224x3). Imagenet also contains 1000 classes while traffic sign dataset contains only 43.

I did implement decaying learning rate because I observed that training was not occurring in the later epochs. I also tried to implement batch normalization, but I did not gain test accuracy gain. As we progress in the course, I would like to implement newer concepts and architectures such as weightnorm, layernorm, DropConnect, Inception, ResNet, etc. Also, I would like to investigate why certain hyperparameters worked in my model. Decaying learning rate worked but why 5e-4 with decaying rate of 0.94 every 100000? I'm not sure. I would also like to investigate other methods against overfitting. I've tried to train my model for more epochs, but the validation accuracy usually went down for much longer training than a few hundred epochs. That indicates that my regularization techniques such as dropout at the last two fully connected layers were not enough. 

After 100 epochs my validation set accuracy reached 0.961 with the test set accuracy of 0.937

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
<img src="/test_images/0.jpg" width="200" height="200">
<img src="/test_images/1.jpg" width="200" height="200">
<img src="/test_images/2.jpg" width="200" height="200">
<img src="/test_images/3.jpg" width="200" height="200">
<img src="/test_images/4.jpg" width="200" height="200">

The images chosen are fairly easy examples. Some have noise due to image labeling, but they are all centered, clear images of signs.

#### 2. Model predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead 		| Turn right ahead								| 
| Ahead only     		| Ahead only 									|
| Stop					| Stop											|
| Road work 			| Road work         							|
| 120 km/h              | 30 km/h                                       |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is within the bounds of test accuracy of 0.937. You would expect the model to predict around 4 to 5 out of 5 images correctly. While predicting 30 km/h vs 120 km/h could be fatal for the self-driving car system, you can see why the model would predict such: round, red circle with 0 at the end, etc.

#### 3. Top 5 Prediction

Because of either the floating point error or a bug in the code, I have not been able to extract the top 5 probabilities. However, using the logit values, I extracted top 5 for each prediction. Correct predictions are the top logit values except the last image

Turn Right Ahead
| Logit             	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  2963.87 			    | Turn Right Ahead								| 
|  616.07  				| Keep Left										|
| -1022.29				| Bumpy road									|
| -1087.41     			| Roundabout mandatory			 				|
| -1184.68			    | Yield             							|


Stop
| Logit                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  3764.96              | Stop                                          | 
|  2610.13              | No vehicles                                   |
|  542.16               | No Entry                                      |
|  445.44               | Speed Limit (20km/h)                          |
|  78.31                | Speed Limit (30km/h)                          |


Ahead only
| Logit                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  4344.70              | Ahead only                                    | 
|  1812.65              | Go straight or right                          |
|  1267.95              | Turn left ahead                               |
|  655.75               | Speed Limit (60km/h)                          |
|  545.63               | Turn right ahead                              |


Road work
| Logit                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  3007.53              | Road work                                     | 
|  1122.13              | Yield                                         |
|  683.81               | Beware of ice/snow                            |
| -479.25               | Bicycles crossing                             |
| -555.93               | Speed limit (80km/h)                          |

Speed limit (120km/h)
| Logit                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  659.89               | Speed limit (20km/h)                          | 
|  120.17               | Vehicles over 3.5 metric tons prohibited      |
| -66.54                | Speed limit (60km/h)                          |
| -209.90               | End of speed limit (80km/h)                   |
| -248.27               | Stop                                          |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


