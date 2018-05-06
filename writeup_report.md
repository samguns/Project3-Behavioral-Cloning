# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_example.jpg "Center lane driving example"
[image2]: ./images/center.jpg "Center"
[image3]: ./images/center-flipped.jpg "Flipped center"
[image4]: ./images/loss_figure.png "MSE Loss Figure"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone_model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone_model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model adopts the same normalization methods that are shown in the lecture slides. And it has pretty much the same architecture as Nvidia’s, the only difference is I added a dropout to mitigate overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone_model.py line 69).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. But I left it configurable along with providing a pre-trained weights. This allows me to tune the model on different data samples in order to enhance predictions of sharp turns specifically.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the center lane driving images and augmented them with LR flipping.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make it memorize lanes and it’s relevant steering angles. Then make a generalized prediction on steering when facing similar lane line situations.

My first step was to use the same network model as Nvidia’s. Since it’s a proven model that works for a real autonomous car, I simply copy that model as my starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but on some training epochs, the validation errors were high and seemed the gap between them were never shortening. I think this implies overfitting.

To combat the overfitting, I tried apply Batch Normalization immediately follows Convolution layer (comment out lines 62-64 in clone_model.py). It increased the training time but I didn’t see any improvements. So I abandoned this approach.

Then I tried to add some dropouts. I only added a 0.5 dropout after the final convolution layer. Luckily, it turned out the model performed better compared to my previous works.

Out of curiosity, I found there’s a provided data samples in the lecture resource tab. I tried my model on this data set and the results were promising, the car drove through the bridge and just slightly missed the sharp corner. I figured maybe adding some more training data would help, and it did. After I combined my collected data with it, and augmented them by LR flipping the center images, the car finally completed one lap of driving autonomously around the track.

#### 2. Final Model Architecture

The final model architecture (clone_model.py lines 58-74) consisted of a convolution neural network with the following layers and layer sizes:_

| Layer | Output Shape |
| ------------- |:-------------:|
| Cropping | 66x320x3 |
| Conv 1 | 31x158x24 |
| Conv 2 | 14x77x36 |
| Conv 3 | 5x37x48 |
| Conv 4 | 3x35x64 |
| Conv 5 | 1x33x64 |
| Dropout (0.5) | 1x33x64 |
| FC 1 | 100 |
| FC 2 | 50 |
| FC 3 | 10 |
| Output | 1 |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one in clock-wise and counter clock-wise order respectively. Here is an example image of center lane driving:

![image1]

To augment the data sat, I flipped images and angles thinking that this would help in training the model. It proved to be effective and efficient, because it simply generates twice amount of datas without spending time on road driving. And the results show that more data helps in mitigating overfitting. Here is an image that has then been flipped:

![image2]
![image3]


After the collection process, I had 12687 data points (mine plus Udacity provided). I then normalized this data by dividing 255 followed by minus 0.5. Before feeding them into network pipeline, I also cropped the upper and bottom unnecessary parts, which finally results in a 66x320x3 shaped image.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was no more than 5 and no less than 2. As evidenced by loss tracing figure below, as well as simulator autonomous driving tests. I found that less training epochs don’t generalize the prediction well and more epochs don’t necessarily yield better results, on the contrary, it makes the model prone to overfit.

![image4]
