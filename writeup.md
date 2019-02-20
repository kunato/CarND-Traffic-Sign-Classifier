# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./markdown_images/random_img.png "Random Dataset Image"
[image2]: ./web_images/1_stop.jpg "Traffic Sign 1"
[image3]: ./web_images/2_stop.jpg "Traffic Sign 2"
[image4]: ./web_images/3_right_of_way.jpg "Traffic Sign 3"
[image5]: ./web_images/4_animal_crossing.jpg "Traffic Sign 4"
[image6]: ./web_images/5_road_work.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python len() to get size of training, validation, test set and I used numpy ndarray .shape to get image shape.
I use ```train['labels'].max() - train['labels'].min() + 1``` to calculate the number of labels in the data set.

* The size of training set is 34799
* The size of the validation set is 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
Showing random image contains in a dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first, I tried convert the images to grayscale but I got better result when training using RGB images. So, I omited grayscale converting step. 

Now, I only did an image normalization to -1.0 to 1.0 by using ```cv2.normalize(img.astype(np.float32), None, -1.0, 1.0)``` on the image data.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24   |	
| RELU					|												|
| Dropout 20%           |                                               |
| Convolution 6x6	    | 1x1 stride, valid padding, outputs 5x5x48     |
| RELU                  |                                               |
| Dropout 20%           |                                               |
| Flatten       		| 5x5x48 -> 1200                                |
| FC                    | 1200 -> 640                                   |
| RELU                  |                                               |
| Dropout 20%           |                                               |
| FC                    | 640 -> 320                                    |
| RELU                  |                                               |
| Dropout 10%           |                                               |
| FC                    | 320 -> 43                                     |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an BATCH_SIZE = 128, EPOCHS = 50 and learning_rate = 0.001 and using AdamOptimizer with softmax_cross_entropy_with_logits as a training function.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.9%
* test set accuracy of 93.6%

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

##### My iterative approach

I first trying more advance architecture such as, concat output from 1st layer conv and 2nd layer conv then follow by 2 layers of FC layer, however, I got quite a bad accuracy, at its best on 10 Epoch. I got only around 87% on the test set. This architecture is more simple but I got better accuracy.
I tuned only the epoch parameter, I try 10 and it gets around 92% on the test set. I tried 50 and then get around 93.8% on the same test set.
I think convolution layer will work well with this problem as it is an image and dropout should reduce the chance to overfitting the problem. I think RGB input should be better than only gray input as a color is also a piece of important information we saw on the traffic sign, so I use RGB image as an input.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The first image might be difficult to classify because of background area is large.
The second image should not be hard to classify as it is similar to the training image.
3rd image is look easy to classify as human because the background was already removed, however, It might be hard for neural network.
4th image should not be difficult to classify but it is also have no background as 3rd image.
5th image should be middle ground for the difficult ratio.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Stop Sign    			| Stop Sign										|
| Right-of-way			| Turn Left Ahead								|
| Wild Animals Crossing	| Wild Animals Crossing			 				|
| Road Work	    		| Road Work      					    		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is quite a bad accuracy compare to 93% of the test set.
I think, a image background might be a reason that neural-network can not classify web_images correctly.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority Road (probability of 1.0), however, on the last run it classify as a stop-sign ? 
I not quite sure why it happen.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority Road   								| 
| 1.58410399e-21 		| Stop Sign									    |
| 1.15398926e-26		| Bicycles Crossing         					|
| 2.86345275e-31	    | No passing by vehicles over 3.5 ...			|
| 0             	    | Speed limit (20 km/h)                         |


For the second image, the model classify the result as Stop sign (probability of 1.0) and It is a stop sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop Sign       								| 
| 7.2803514e-09 		| Yield							                |
| 1.9222478e-16			| Road work         			                |
| 7.6348943e-17	      	| Priority Road	                    		    |
| 4.3678412e-17		    | Bumpy road                                    |


For 3rd image, the model classify the image as No passing (100%)
, which is wrong. The actual result (Right-of-way at the next intersection) is also in the top 5 result.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| No passing 							        | 
| 4.9984776e-03 		| Turn left ahead							    |
| 5.7042803e-04			| Roundabout mandatory         					|
| 3.4516630e-14	      	| Slippery road			                        |
| 1.4723741e-21		    | Right-of-way at the next intersection         |


For 4th image, 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop Sign       								| 
| 8.2245285e-29 		| Animal crossing				                |
| 1.4610156e-29			| Double curve        			                |
| 3.8137516e-34	      	| General caution                    		    |
| 2.8262647e-37		    | Road work                                     |


For 5th image,


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work       								| 
| 3.14701571e-21 		| Animal crossing				                |
| 2.27639822e-21		| Bumpy road        			                |
| 4.20068679e-25	    | Traffic signal                      		    |
| 1.00364074e-25		| Stop                                          |




