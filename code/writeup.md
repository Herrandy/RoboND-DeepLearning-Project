## Follow Me

---

[Model Weights](https://github.com/JafarAbdi/RoboND-DeepLearning-Project/blob/master/data/weights/model_weights)

[Jupyter Notebook](https://github.com/JafarAbdi/RoboND-DeepLearning-Project/blob/master/code/model_training.ipynb)

[HTML version of the Notebook](https://github.com/JafarAbdi/RoboND-DeepLearning-Project/blob/master/model_training.html)

[//]: # (Image References)

[network]: ../data/imgs/network.png
[conv]: ../data/imgs/1x1conv.png

### Introduction:

The task was to build a segmentation network, train it and validate it. End product should be a drone tracker which follows a hero target.

In addition to above, optionally one could gather his/her own training data and use it in the training and validation process but due to the lack of time and 
unable to get AWS working this is left as a future project.

### Network architecture:

The network is described in the figure below:

![network]


The network architecture is fully-convolutional neural network (FCN) which consist of five different encoders/decoders with the depth size of 16, 32, 64, 128, 256. We are using a stride of 2 which means that
the encoder subsamples the images width and height by factor of 2 and the decoder uses bilinear upsampling again with factor of 2. This means we will end-up with same size input and output (160x160x3).
Between the encoder and decoder is a 1x1 convolution layer having depth of 512. Idea of the 1x1 convolutional layer is to preserve the spatial information.







### 1x1 Convolutional layer

![conv]
The 1x1 convolution simply means that each activation in input layer is multiplied by a single value and no information about neighbouring coordinates is used.
The 1x1 convolutional layer has dual purpose: they are used mainly for dimension reduction which leads to 
faster computation and one can then make the network more deeper. In addition we can add more non-linearity by having ReLU immediately after every 1x1 convolution.

The size of 1x1x512 convolution layer is used in the task. This will change the dimensions of input layer (5, 5, 256) ==> 
(5, 5, 512). 


### Hyperparameters

Learning rate: Describes the rate how the network adjust the weight based on the training data. 
If too big value is used the network can easily overshoot and even became unstable. If the value is too small the search of the optimal values for the weights
can take lots of time.

Batch size: Descibes the number of samples that are propagated through the network.

Number of epochs: Number of times the whole data set is went throught the network. For instance, if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

Steps per epoch: Number of batches of training images that go through the network in 1 epoch. The value could be selected so that 
the entire dataset is propagated throught the network: number_of_training_data / batch_size. If the data set is huge, this could reduce the time for training.

Validation steps: Number of iterations (batches of samples) used for validating 


### 6- Final results

the video below show the final results 

[Final Results](https://www.youtube.com/watch?v=QlZK7eJRojE)

### 7-Future Enhacement

1- in the training curve the final epoch's traning loss = 0.0115 and validation loss = 0.0341 overfitting like ==> adding more data can improve this

2- trying one of the existed architecture (for example [LINK](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review) )

3- the video show that the drone follow the hero for the whole recording time but it takes sometimes >= 5 min to catch him, I think doing the previous two steps can improve this.

I don't think this model will works well for following small object (e.g. cats, dogs) specially from long distance and some angles, but it should works well (maybe not the same final score as this) for larger object (e.g. car, horses)

