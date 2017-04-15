import csv
import cv2
import numpy as np

lines = []
images = []
measurements = []
correction = 0.1 # obtained by tuning
cnt = 0

def read_image_info(lines, folder_name):
    with open(folder_name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def read_image_and_steering(images, measurements, lines, folder_name):
    # update the images and measurements based on
    # the info from driving_log.csv
    cnt = len(images)
    for line in lines[cnt:]:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = folder_name + '/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)

        # adjusted steering measurements for the side camera images
        if cnt % 3 == 0: # center image
            measurements.append(float(line[3]))
        elif cnt % 3 == 1: # left image
            measurements.append(float(line[3]) + correction)
        elif cnt % 3 == 2: # right image
            measurements.append(float(line[3]) - correction)
        cnt += 1
    return images, measurements

# read in the image info in the folder Data
# lines = read_image_info(lines, 'Data')
# images, measurements = read_image_and_steering(images, measurements, lines, 'Data')
lines = read_image_info(lines, 'Data1')
images, measurements = read_image_and_steering(images, measurements, lines, 'Data1')

# augment data by flipping the existing iamges
augmented_images, augmented_measurements =[], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

input_shape = X_train[0].shape
print('Input shape is {:}'.format(input_shape))

# Build a LeNet like neural network
model = Sequential()
# cropping the images to remove useless pixels
# TO DO: cropped size to be tuned significantly
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((7,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
