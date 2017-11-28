## Follow Me

---

[//]: # (Image References)

[network]: ../data/imgs/network.png
[conv]: ../data/imgs/1x1conv.png
[training]: ../data/imgs/training.png

### Introduction:

The task was to build a segmentation network, train and validate it. End product should be a drone tracker which follows a hero target. 
The algorithm should categorize each pixel in the image either into hero, random human or background pixel.

In addition to above, optionally one could gather his/her own training data and use it in the training and validation process but due to the lack of time and 
unable to get AWS working this is left as a future project.

### Network architecture:

The network is described in the figure below:

![network]

The network architecture is fully-convolutional neural network (FCN) which consist of five different encoders/decoders with the depth size of 16, 32, 64, 128, 256. 
I am using a stride of 2 which means that the encoder subsamples the images width and height by factor of 2 and the decoder uses bilinear upsampling again with factor of 2. 
This means we will end-up with same size input and output (160x160x3).

The encoder is a series of convolutional layer where a sliding window (convolutional kerner) is slide through the image. 
Idea of the encode is to extract features from the image and translate them to compressed and more abstract representation.
In the encoder part we are using batch normalization which advantages are for instance: 1) network train faster, 2) allows higher learning rate, 
3) simplifies the creation of deeper networks and 4) Provides a bit of regularization
The decoder upsamples the encoded image to have the same size as the input image where each of the pixels classified.
In decoder separable convolution layers are used to extract some more spatial information from prior layers.

The network uses skip connections which allow the network to use information from multiple resolution scales.
In the encoding part of the network we will lose some finer details of the images so using the skip connection we can retain some of this information.
This should lead to more precise segmentation of the objects on the image.

Between the encoder and decoder is a 1x1 convolution layer having depth of 512. 
Idea of the 1x1 convolutional layer is to preserve the spatial information. 
Other advantage for instance over the fully connected layer is that the network can take as an input arbitary size images.

### 1x1 Convolutional layer

![conv]

The 1x1 convolution simply means that each activation in input layer is multiplied by a single value and no information about neighbouring coordinates is used.
The 1x1 convolutional layer has dual purpose: they are used mainly for dimension reduction which leads to 
faster computation and one can then make the network more deeper. In addition we can add more non-linearity by having ReLU immediately after every 1x1 convolution.
However 1x1 convolution is not always used to shrink the output depth, it might also useful to keep it the size of the depth same or even increase it.

The size of 1x1x512 convolution layer is used in the task. This will change the dimensions of input layer (5, 5, 256) ==> 
(5, 5, 512). This actually does not reduce the size of detph but increases it. 
During the experiments it was noted to give better results which might be due to that fact we are trying to learn more complex functions.



### Hyperparameters

Learning rate: Describes the rate how the network adjust the weight based on the training data. 
If too big value is used the network can easily overshoot and even became unstable. If the value is too small the search of the optimal values for the weights
can take lots of time.

Batch size: Descibes the number of samples that are propagated through the network.

Number of epochs: Number of times the whole data set is went throught the network. For instance, if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

Steps per epoch: Number of batches of training images that go through the network in 1 epoch. The value could be selected so that 
the entire dataset is propagated throught the network: number_of_training_data / batch_size. If the data set is huge, this could reduce the time for training.

Validation steps: Number of iterations (batches of samples) used for validating 


### Results

Following hyperparameters were used in the training process:

```python
learning_rate = 0.01
batch_size = 16
num_epochs = 70
steps_per_epoch = 4131 // 16
validation_steps = 50
workers = 4
```
Using the above parameters final score of 0.436 was achieved.

The learning rate was relatively high but it still gave good results. 
Maximum possible batch size was 16, otherwise my graphic card would run out of memory.
Steps per epoch was chosen so that entire dataset was propagated during the training process.

Below is the result of the training:

![training]

[Model Weights](https://github.com/Herrandy/RoboND-DeepLearning-Project/tree/master/data/weights/model_weights)

[Jupyter Notebook](https://github.com/Herrandy/RoboND-DeepLearning-Project/tree/master/code/model_training.ipynb)

[HTML version of the Notebook](https://github.com/Herrandy/RoboND-DeepLearning-Project/tree/master/model_training.html)


### Future Enhancements

Using the current weights of the network we are only capable of tracking reddish human targets. 
The network structure should be appropriate for tracking different objects (such as cats and dogs) but we would need 

Following the discussion in slack people are also achieving good result using a smaller number of layers (3 or 4) so in the future
it would be interesting to try shallower networks.

To make the training process more enjoyable I will try to set-up AWS. This would probably allow to test bigger batch sizes and it would be interesting to see their 
affect to the performance.