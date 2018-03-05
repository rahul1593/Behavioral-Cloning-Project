import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
#from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, Activation, Cropping2D, Lambda
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Cropping2D, Lambda

# Config: Modify when necessary
data_root = './data'
batch_size = 64
EPOCHS = 7


# read the driving log csv file
samples = []
csvfile = open(data_root+"/driving_log.csv")
reader = csv.reader(csvfile)
stear_correction = 0.3

# merge all camera data into front camera data, adjust stear angle
f = False
for line in reader:
    if not f:    # skip the first line, which does not contain any data
        f = True
        continue
    steer = float(line[3])
    samples.append([line[0], steer, False])
    samples.append([line[1], steer+stear_correction, False])
    samples.append([line[2], steer-stear_correction, False])
    if -0.01 <= steer and steer >= 0.01:    # in case car is turning, also add the flipped version
        samples.append([line[0], -steer, True])
        samples.append([line[1], -steer-stear_correction, True])
        samples.append([line[2], -steer+stear_correction, True])
print("Log read")
csvfile.close()
# shuffle the samples and split the training and validation sets
samples = shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("Total training samples:", len(train_samples))
print("Total validation samples:", len(validation_samples))

# display some stats
ipath = data_root+'/IMG/'+train_samples[0][0].split('/')[-1]
print(ipath)
image = cv2.imread(ipath)
print("Sample dimensions:", image.shape)
timg = cv2.flip(image, flipCode=1)
plt.imshow(image)
plt.show()
plt.imshow(timg)
plt.show()
### plot histogram to see steer data distribution ###
#train_angles = []
#for v in train_samples:
#    train_angles.append(v[1])
#plt.hist(train_angles, 50, normed=1, facecolor='green', alpha=0.75)
#plt.xlabel('Steer Angle')
#plt.ylabel('Frequency')
#plt.show()
#############################

# function to generate data for the batch
def generator(samples, batch_size=64):
    pdir = data_root+'/IMG/'
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = pdir+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                if batch_sample[2]:    # if True then flip the image
                    center_image = cv2.flip(center_image, flipCode=1)
                center_angle = batch_sample[1]
                images.append(center_image)
                angles.append(center_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

print("Read images")

# model
model = Sequential()
# Crop the image to remove unwanted area from the image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
# Convert the image to grayscale, input format is BGR
model.add(Lambda(lambda x:(0.163 * x[:,:,:,:1]) + (0.4870 * x[:,:,:,1:2]) + (0.35 * x[:,:,:,-1:])))
# Normalize the data in image
model.add(Lambda(lambda x:x/255))

#model.add(Convolution2D(64, 7, 7, border_mode='same'))
model.add(Conv2D(64, (7, 7), padding="same"))        # input: 90x320x1, output: 90x320x64
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), padding='same'))       # input: 90x320x64, output: 45x160x64

#model.add(Convolution2D(32, 5, 5, border_mode='valid'))
model.add(Conv2D(32, (5, 5), padding="valid"))        # input: 45x160x64, output: 41x156x32
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), padding='same'))       # input: 41x156x32, output: 19x78x32

#model.add(Convolution2D(24, 3, 3, border_mode='valid'))
model.add(Conv2D(24, (3, 3), padding="valid"))        # input: 19x78x32, output: 17x76x24
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), padding='same'))       # input: 17x76x24, output: 9x38x24

model.add(Conv2D(16, (3, 3), padding="valid"))        # input: 9x38x24, output: 7x36x16
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), padding='same'))       # input: 7x36x16, output: 4x18x16

model.add(Flatten())            # input: 4x18x16, output: 1152
model.add(Dropout(0.3))

model.add(Dense(516))           # input: 1152, output: 516
model.add(Activation('elu'))

model.add(Dropout(0.2))

model.add(Dense(1))             # input: 516, output: 1
print("Training")

# Train the model
model.compile(loss="mse", optimizer="adam")
#model.fit_generator(train_generator, samples_per_epoch= \
#            len(train_samples), validation_data=validation_generator, \
#            nb_val_samples=len(validation_samples), nb_epoch=7, verbose=1)
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch_size,
            validation_data=validation_generator,
            validation_steps=len(validation_samples)/batch_size, epochs=EPOCHS, verbose=1)
print("Trained")

# Save the model
model.save('model.h5')
print("Model saved.")
