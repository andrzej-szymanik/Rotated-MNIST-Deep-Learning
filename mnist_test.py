#classifier = load_model('my_neural_network.h5') 

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('check/6.jpg')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis =0)
result = classifier.predict(test_image)
 
