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

---
### Writeup

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


#### 3. Training Approach
To train the model, I used a GTX 1060 I recently bought for this course. Training the data with the architecture mentioned above took about 5 seconds per epoch. I trained to 100 epochs which took about 8 minutes to complete. While I did not do any grid search for hyperparameters, I found that learning rate of 5e-4 was suitable, as it seemed more stable than 1e-3 and quicker to learn than 1e-4. Batch size used was 128 as given.

Initially, I chose LeNet as given in the project, but I wanted to try to implement more of a recent model architecture that was simple enough to implement on raw tensorflow from scratch, thus VGG net was chosen. Compared to the actual VGG net, this architecture skips a few steps such as a couple of conv layers in the middle and the last fully connected layer. While training, I saw that the increase in model complexity did not necessarily increase in test accuracy, perhaps due to the traffic sign images containing much less information than the Imagenet (32x32x3 vs 224x224x3), and Imagenet containing 1000 classes while traffic sign dataset containing only 43.

I did implement decaying learning rate because I observed that training was not occurring in the later epochs. I also tried to implement batch normalization, but I did not gain test accuracy gain perhaps due to my implementation error. As we progress in the course, I would like to implement newer concepts and architectures such as weightnorm, layernorm, DropConnect, Inception, ResNet, etc. Also, I would like to investigate why certain hyperparameters worked in my model. Decaying learning rate worked but why 5e-4 with decaying rate of 0.94 every 100000? I'm not sure. I would also like to investigate other methods against overfitting. I've tried to train my model for more epochs, but the validation accuracy usually went down for much longer training than a few hundred epochs. That indicates that my regularization techniques such as dropout at the last two fully connected layers were not enough.

After 100 epochs my validation set accuracy reached 0.969 with the test set accuracy of 0.945

### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

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
| 120 km/h              | 30 km/h                                       |
| Road work             | Road work                                     |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is within the bounds of test accuracy of 0.945. You would expect the model to predict around 4 to 5 out of 5 images correctly. While predicting 30 km/h vs 120 km/h could be fatal for the self-driving car system, you can see why the model would predict such: round, red circle with a 0 at the end, etc.

#### 3. Top 5 Prediction

Because of either the floating point error or a bug in the code (TODO: Fix top 5 prob.), I have not been able to extract the top 5 probabilities. However, using the logit values, I extracted top 5 for each prediction. Correct predictions are the top logit values except the 120 km/h image. As you can see in the logit values, there is usually a large difference between the top prediction and the second highest, except the speed limit which the model predicted incorrectly.

Turn Right Ahead

| Logit             	|     Prediction	        					| 
|:---------------------:|:--------------------------------------------- | 
|  2391.62 			    | Turn Right Ahead								| 
|  950.83 				| Ahead only									|
|  341.95				| Go straight or right							|
|  62.56     			| Turn left ahead			 				    |
| -337.75			    | Keep left             						|

Ahead only

| Logit                 |     Prediction                                | 
|:---------------------:|:--------------------------------------------- | 
|  4620.69              | Ahead only                                    | 
|  1384.34              | Speed limit (60km/h)                          |
| -160.42               | Go straight or left                           |
| -217.22               | Turn left ahead                               |
| -420.67               | Turn right ahead                              |

Stop

| Logit                 |     Prediction                                | 
|:---------------------:|:--------------------------------------------- | 
|  1623.36              | Stop                                          | 
|  702.17               | End of all speed and passing limits           |
|  184.71               | No entry                                      |
|  99.20                | Speed Limit (60km/h)                          |
|  10.52                | No vehicles                                   |



Speed limit (120km/h)

| Logit                 |     Prediction                                | 
|:---------------------:|:--------------------------------------------- | 
|  238.98               | Road work                                     | 
|  141.89               | Stop                                          |
|  41.54                | No vehicles                                   |
| -29.62                | Speed limit (50km/h)                          |
| -33.98                | Yield                                         |


Road work

| Logit                 |     Prediction                                | 
|:---------------------:|:--------------------------------------------- | 
|  1189.90              | Road work                                     | 
|  570.29               | Wild animals crossing                         |
| -77.31                | Bumpy road                                    |
| -19.94                | Bicycles crossing                             |
| -338.29               | Yield                                         |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
#### TODO
