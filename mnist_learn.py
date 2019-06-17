from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, 3, 3, border_mode='same',
                             input_shape=(28, 28, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=10, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train_images',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'test_images',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical')

classifier.fit_generator(training_set, steps_per_epoch=22000, epochs=25, validation_data=test_set,
                         validation_steps=2000)

classifier.save('project.h5')  # creates a HDF5 file 'my_model.h5'
