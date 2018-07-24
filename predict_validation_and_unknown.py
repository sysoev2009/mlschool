from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import re, os, sys

img_width, img_height = 927, 383

f = open('class_dictionary.txt')
lines = f.readline()
class_dictionary = ['None'] * int(lines)
i = 0
while i < int(lines):
	line = re.sub(r'\s+', ' ', f.readline())
	line = line.split(' ')
	class_dictionary[int(line[1])] = line[0]
	i += 1
f.close()

model_json_file = open("model.json", "r")
model_json = model_json_file.read()
model_json_file.close()
model = model_from_json(model_json)

weights = os.listdir('weights')
max_val_acc = 0
n = 0
i = 0
while i < len(weights):
	val_acc = weights[i].split(' .')[0]
	val_acc = float(val_acc.split(' ')[3])
	if val_acc > max_val_acc:
		max_val_acc = val_acc
		n = i
	i += 1

model.load_weights('weights/'+weights[n])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

f = open('result.txt', 'w')

l0 = os.listdir('validation')
i = 0
while i < len(l0):
	l1 = os.listdir('validation/'+l0[i])
	j = 0
	while j < len(l1):
		filename = l1[j].replace('png', 'wav')
		spectrogram = image.load_img('validation/'+l0[i]+'/'+l1[j], target_size=(img_width, img_height, 3), grayscale=False)
		x = image.img_to_array(spectrogram)
		x /= 255
		x = np.expand_dims(x, axis=0)
		prediction_class = model.predict_classes(x)
		class_probability = model.predict(x)
		class_probability = np.round(class_probability[0][prediction_class], 3)
		f.writelines(filename +' ' + str("%.3f" % class_probability) + ' ' + class_dictionary[prediction_class[0]] + '\n')
		j += 1
	i += 1

spectrograms = os.listdir('test')
i = 0
while i < len(spectrograms):
	spectrogram = image.load_img('test/'+spectrograms[i], target_size=(img_width, img_height, 3), grayscale=False)
	x = image.img_to_array(spectrogram)
	x /= 255
	x = np.expand_dims(x, axis=0)
	class_probability = model.predict(x)
	prediction_class = model.predict_classes(x)
	class_probability = np.round(class_probability[0][prediction_class], 3)
	spectrograms[i] = spectrograms[i].replace('png', 'wav')
	f.writelines(spectrograms[i] + ' ' + str("%.3f" % class_probability) + ' ' + class_dictionary[prediction_class[0]] + '\n')
	i += 1
print()
f.close()
