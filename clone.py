import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

lines = [] # list of image names
images = [] # list of images
measurements = [] # list of steering angles
correction = 0.1 # obtained by tuning

def read_image_info(lines, folder_name):
    """read the names of images in a given folder"""
    with open(folder_name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def read_image_and_steering(images, measurements, lines, folder_name):
    """update the images and measurements based on
        the info from driving_log.csv"""
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

# specify the list of folders to read data
data_list = ['Data_1st_lap','Data_2nd_lap', 'Data_3rd_lap',\
        'Data_correct_direction_1', 'Data_correct_direction_2', 'Data_correct_direction_3',\
       'Data_turn_lap_1','Data_turn_lap_2', 'Data_track_2',\
       ]
# data_list = ['Data_1st_lap','Data_2nd_lap', 'Data_3rd_lap', \
#        'Data_correct_direction_1', 'Data_correct_direction_2']

for data in data_list:
    lines = read_image_info(lines, data)
    images, measurements = read_image_and_steering(images, measurements, lines, data)

# augment data by flipping the existing iamges and steering angles
augmented_images, augmented_measurements =[], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)

# treat the augmented images/measurements as the entire data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

input_shape = X_train[0].shape
print('Input shape is {:}'.format(input_shape))

# Build a Nvidia like neural network
model = Sequential()
# use Lambda function to normalize the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
# cropping the images to remove useless pixels
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)

model.save('model.h5')
