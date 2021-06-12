import csv
import cv2
import numpy as np

lines = []


with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/'+filename
        image = cv2.imread(current_path)
        images.append(cv2.flip(image,1))
        measurement = float(line[3])
        measurements.append(measurement*-1.0)

X_train = np.array(images)
Y_train = np.array(measurements)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
model.summary() 
  
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,Y_train,validation_split=0.2,shuffle=True, epochs  = 10, verbose=1)

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')