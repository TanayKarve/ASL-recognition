# Part 1 - Building the CNN
import matplotlib.pyplot as plt
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import Dense,Dropout
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 29, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('asl-alphabet//train//asl_alphabet_train', #Might throw filenotfound error. Replace                                                                                              #with actual
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('asl-alphabet//test//asl_alphabet_test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

call=ModelCheckpoint('asl.{epoch:02d}.hdf5',monitor='acc',verbose=1,save_best_only=True)
cb=[call]
classifier=load_model('asl.h5')
#try:
#    history=classifier.fit_generator(training_set,
#	                         steps_per_epoch = 8000,
#	                         epochs = 20,
#	                         validation_data = test_set,
#	                         callbacks=cb,
#	                         validation_steps = 2000)
#except Exception as e:
#	print(e)
#	classifier.save('asl.h5')

#classifier.save('asl.h5')
#plt.plot(history.history['acc'],linewidth=3)
#plt.plot(history.history['val_acc'],linewidth=3)
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()