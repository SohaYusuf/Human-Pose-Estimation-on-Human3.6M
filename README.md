# Human-Pose-Estimation-on-Human3.6M

In this assignment, we use Human3.6M dataset to train a neural network. Training dataset has 5964 video
samples and test dataset has 1368 video samples. Each video sample has 8 frames each having 224 x
224 x 3 image and we train the model for regression learning with mean-squared error loss. The model
architecture contains CNN (ResNet), LSTM and MLP.

Hyper-parameters used:
Learning rate = 0.0001
Batch size = 12
Epochs = 30
Optimizer = Adam

Training loss and testing loss decrease with the number of epochs. Model always performs better on training
data than testing data. In our experiment, over fitting is not happening because both loss curves continue to
decrease. At 30 epochs, training loss reaches 0.0041 and test loss reaches 0.0068.

![image](https://user-images.githubusercontent.com/102180459/167032382-43db8655-4738-4028-9d5b-89e7b70d3cf5.png)

Mean MPJPE is calculated at the end of each epoch and plotted for training and testing data. Both testing
and training MPJPE curves decrease with the number of epochs. MPJPE reaches less than 150 mm on both
training and testing data. At 30 epochs, MPJPE reaches 140.15 mm on training data and 128.09 mm on test
data.

![image](https://user-images.githubusercontent.com/102180459/167032507-5dcc1ed7-7757-471b-bb16-5b45a738afe2.png)
