from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import pathlib, shutil, os

train_dir = 'train'
val_dir = 'validation'
img_width, img_height = 927, 383
input_shape = (img_width, img_height, 3)
epochs = 15
batch_size = 16
nb_train_samples = 22612
nb_validation_samples = 473

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(72, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_json = model.to_json()
model_json_file = open("model.json", "w")
model_json_file.write(model_json)

shutil.rmtree('weights', ignore_errors=True)
pathlib.Path('weights').mkdir(parents=True, exist_ok=True)
checkpoint = []
checkpoint.append(ModelCheckpoint('weights/epoch {epoch:02d} val_acc {val_acc:.4f} .hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1))

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

class_dictionary = train_generator.class_indices
f = open('class_dictionary.txt', 'w')
f.writelines(str(len(class_dictionary)) + '\n')
for label in class_dictionary:
	f.writelines(label + ' ' + str(class_dictionary[label]) + '\n')
f.close()

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
	callbacks=checkpoint)
	
max_val_acc = 0
weights = os.listdir('weights')
i = 0
while i < len(weights):
	val_acc = weights[i].split(' .')[0]
	val_acc = float(val_acc.split(' ')[3])
	if val_acc > max_val_acc:
		max_val_acc = val_acc
	i += 1
print('\nMax validation accuracy: ' + str(max_val_acc) + '\n')