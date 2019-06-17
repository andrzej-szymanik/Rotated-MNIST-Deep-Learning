from keras.models import load_model
import os
from keras.preprocessing import image
classifier = load_model('/home/andrzej/Applications/MNIST/project.h5') 
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

import numpy as np
success = 0
number = 9
i = 0
images = []
for img in os.listdir(os.getcwd()):
    img = image.load_img(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
images = np.vstack(images)
classes = classifier.predict_classes(images,batch_size = 10)
for i in range (0,classes.size) :
    if classes[i]  == number:
        success = success + 1
success = success / classes.size * 100
print(classes)
print(str(success) + ' %')

