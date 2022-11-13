# make a prediction for a new image.
from numpy import argmax
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	#img = load_image('1.png')
	#img = load_image('2.png')
	#img = load_image('3.png')
	#img = load_image('4.png')
	#img = load_image('5.png')
	#img = load_image('6.png')
	#img = load_image('7.png')
	#img = load_image('8.png')
	#img = load_image('9.png')
	img = load_image('10.png')
	#img = load_image('test_image.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)

# entry point, run the example
run_example()
