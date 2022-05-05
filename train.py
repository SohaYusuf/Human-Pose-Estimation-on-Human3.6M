"""
Author: SOHA YUSUF 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow.keras.layers import Add, Conv2D, Dropout, Flatten, LSTM, MaxPooling2D, TimeDistributed, \
    BatchNormalization, Reshape


# load the training data
train_data = np.load('data/videoframes_clips_train.npy')
train_label = np.load('data/joint_3d_clips_train.npy')

# load the testing data
test_data = np.load('data/videoframes_clips_valid.npy')
test_label = np.load('data/joint_3d_clips_valid.npy')

# print the shapes 
print(f"train_data shape: {train_data.shape}")    # (5964, 8, 224, 224, 3)
print(f"train_label shape: {train_label.shape}")    # (5964, 8, 17, 3)
print(f"test_data shape: {test_data.shape}")     # (1368, 8, 224, 224, 3)
print(f"test_label shape: {test_label.shape}")     # (1368, 8, 17, 3)


# build dataset
# initialize batch size
batch_size = 12
tf.debugging.set_log_device_placement(True)
with tf.device('/CPU:0'):
    # build training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_label)).shuffle(buffer_size=10).batch(batch_size)
    # create a map for training dataset
    train_dataset = (train_dataset.map(lambda x, y:
                                       (tf.divide(tf.cast(x, tf.float32), 255.0),tf.cast(y, tf.float32))))
    # build test dataset
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(batch_size))
    # create a map for test dataset
    test_dataset = (test_dataset.map(lambda x, y:
                                     (tf.divide(tf.cast(x, tf.float32), 255.0),tf.cast(y, tf.float32))))

# print the size of datasets
with tf.device('/device:GPU:2'):
    print(len(train_dataset))
    print(len(test_dataset))

# define model architecture
def my_model():
    
    input_shape = (8,224,224,3)
    img = tf.keras.Input(shape=input_shape)
    
    X_shortcut = img
    
    # (8,224,224,3)
    conv1 = TimeDistributed(Conv2D(32, (5, 5), strides=1, padding='valid', activation=tf.nn.relu))(img)
    B1 = TimeDistributed(BatchNormalization())(conv1)
    M1 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B1)
    D1 = TimeDistributed(Dropout(0.25))(M1)
    # (8,110,110,32)
    conv2 = TimeDistributed(Conv2D(64, (5, 5), strides=1, padding='valid', activation=tf.nn.relu))(D1)
    B2 = TimeDistributed(BatchNormalization())(conv2)
    M2 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B2)
    D2 = TimeDistributed(Dropout(0.25))(M2)
    # (8,53,53,64)
    
    # skip connection 1 
    X_shortcut1 = TimeDistributed(Conv2D(64, (16, 16), strides=4, padding='valid',activation=tf.nn.relu))(X_shortcut)
    X_shortcut1 = TimeDistributed(BatchNormalization())(X_shortcut1)
    skip1 = TimeDistributed(Add())([D2, X_shortcut1])   # (8,53,53,64)
    # end
    
    # (8,53,53,64)
    conv3 = TimeDistributed(Conv2D(128, (5, 5), strides=1, padding='valid', activation=tf.nn.relu))(skip1)
    B3 = TimeDistributed(BatchNormalization())(conv3)
    M3 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B3)
    D3 = TimeDistributed(Dropout(0.25))(M3)
    # (8,24,24,128)
    conv4 = TimeDistributed(Conv2D(256, (5, 5), strides=1, padding='valid', activation=tf.nn.relu))(D3)
    B4 = TimeDistributed(BatchNormalization())(conv4)
    M4 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B4)
    D4 = TimeDistributed(Dropout(0.25))(M4)
    # (8,10,10,256)
    
    # skip connection 2 
    X_shortcut2 = TimeDistributed(Conv2D(256, (8, 8), strides=5, padding='valid',activation=tf.nn.relu))(skip1)
    X_shortcut2 = TimeDistributed(BatchNormalization())(X_shortcut2)
    skip2 = TimeDistributed(Add())([D4, X_shortcut2])   # (8,10,10,256)
    # end

    # (8,10,10,256)
    conv7 = TimeDistributed(Conv2D(512, (3,3), strides=1, padding='valid', activation=tf.nn.relu))(skip2)
    B7 = TimeDistributed(BatchNormalization())(conv7)
    M7 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B7)
    D7 = TimeDistributed(Dropout(0.25))(M7)
    # (8,4,4,512)
    conv8 = TimeDistributed(Conv2D(1024, (1, 1), strides=1, padding='valid', activation=tf.nn.relu))(D7)
    B8 = TimeDistributed(BatchNormalization())(conv8)
    M8 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B8)
    D8 = TimeDistributed(Dropout(0.25))(M8)
    # (8,2,2,1024)
    
    # skip connection 3 
    X_shortcut3 = TimeDistributed(Conv2D(1024, (2,2), strides=8, padding='valid',activation=tf.nn.relu))(skip2)
    X_shortcut3 = TimeDistributed(BatchNormalization())(X_shortcut3)
    skip3 = TimeDistributed(Add())([D8, X_shortcut3])     # (8,2,2,1024)
    # end
    
    # (8,2,2,1024)
    conv5 = TimeDistributed(Conv2D(2048, (2, 2), strides=1, padding='same', activation=tf.nn.relu))(skip3)
    B5 = TimeDistributed(BatchNormalization())(conv5)
    M5 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(B5)
    D5 = TimeDistributed(Dropout(0.3))(M5)
    # (8,1,1,2048)
    conv6 = TimeDistributed(Conv2D(4096, (2, 2), strides=1, padding='same',activation=tf.nn.relu))(D5)
    B6 = TimeDistributed(BatchNormalization())(conv6)
    # (8,1,1,4096)
    
    F1 = Flatten()(B6)
    R1 = Reshape((8, 4096))(F1)

    # LSTM layer
    L1 = LSTM(units=1024, activation='tanh', return_sequences=True, dropout=0.3)(R1)

    # MLP with 51 units
    Den3 = TimeDistributed(tf.keras.layers.Dense(51))(L1)
    output = tf.keras.layers.Reshape((8, 17, 3))(Den3)
    
    model = tf.keras.Model(inputs=img, outputs=output)
    return model


model = my_model()
model.summary()


# hyper parameters
lr = 0.0001
EPOCHS = 30

# initialize optimizer, loss function and evaluation metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='mean_squared_error',
    metrics=['acc']
)


# build a class for callbacks for calculating MPJPE
MPJPE_test_list = []
MPJPE_train_list = []
temp_train = []
temp_test = []
# MPJPE
class Metrics(Callback):
    def __init__(self, TEST_DATA, TRAIN_DATA):
        super(Callback, self).__init__()
        self.TRAIN_DATA = TRAIN_DATA
        self.TEST_DATA = TEST_DATA
        
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        # compute MPJPE for training data
        for x_train,y_train in self.TRAIN_DATA:
            predict_train = model.predict(x_train)
            MPJPE_train = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((y_train - predict_train), axis=3)) * 1000
            temp_train.append(MPJPE_train)
        # store mean MPJPE for plotting
        temp_mean_train = tf.reduce_mean(temp_train)
        MPJPE_train_list.append(temp_mean_train)
        print('MPJPE Training: ', temp_mean_train)
        # compute MPJPE for test data
        for x_test,y_test in self.TEST_DATA:
            predict_test = model.predict(x_test)
            MPJPE_test = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((y_test - predict_test), axis=3)) * 1000
            temp_test.append(MPJPE_test)
        # store mean MPJPE for plotting
        temp_mean_test = tf.reduce_mean(temp_test)
        MPJPE_test_list.append(temp_mean_test)
        print('MPJPE Testing: ', temp_mean_test)
            
        return



my_metrics = Metrics(TEST_DATA=test_dataset, TRAIN_DATA=train_dataset)

# train the model
with tf.device('/device:GPU:2'):
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, shuffle=True, callbacks=[my_metrics])


# final loss , accuracy and MPJPE for training data
final_loss_train = history.history['loss'][-1]
final_acc_train = history.history['acc'][-1]
final_MPJPE_train = MPJPE_train_list[-1]
# final loss , accuracy and MPJPE for testing data
final_loss_test = history.history['val_loss'][-1]
final_loss_test = history.history['val_acc'][-1]
final_MPJPE_test = MPJPE_test_list[-1]

print('Performance on training data: \n')
print('Final Training Loss = ',final_loss_train)
print('Final Training Accuracy = ',final_acc_train)
print('Final Training MPJPE = ',final_MPJPE_train.numpy())
print('Performance on testing data: \n')
print('Final Test Loss = ',final_loss_test)
print('Final Test Accuracy = ',final_loss_test)
print('Final Test MPJPE = ',final_MPJPE_test.numpy())



# plot loss curves
loss_train = history.history['loss']
loss_test = history.history['val_loss']

plt.figure(1)
plt.figure(figsize=(8, 8))
plt.plot(loss_train, label=f'Training Loss (a={lr})')
plt.plot(loss_test, label=f'Testing Loss (a={lr})')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid()
#plt.yscale('log')
plt.xlabel('# epochs')
plt.show()



# plot  accuracy curves
acc_train = history.history['acc']
acc_test = history.history['val_acc']

plt.figure(2)
plt.figure(figsize=(8, 8))
plt.plot(acc_train, label=f'Training accuracy (a={lr})')
plt.plot(acc_test, label=f'Testing accuracy (a={lr})')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.grid()
#plt.yscale('log')
plt.xlabel('# epochs')
plt.show()



# plot MPJE curves
plt.figure(3)
plt.figure(figsize=(8, 8))
plt.plot(MPJPE_train_list, label=f'Training MPJPE (a={lr})')
plt.plot(MPJPE_test_list, label=f'Testing MPJPE (a={lr})')
plt.legend(loc='upper right')
plt.ylabel('MPJPE')
plt.title('MPJPE Curve')
#plt.yscale('log')
plt.xlabel('# epoch')
plt.show()



model.save_weights('model_weights')
model.save("trained_model")
model.save("trained_model.h5")
