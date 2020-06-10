from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pickle

# MODEL BUILDING

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def plot_history(history): #plotting accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Setting-up BASE_MODEL

IMG_HEIGHT = 211 # 1689
IMG_WIDTH = 80 # 646

base_model = InceptionV3(weights='imagenet', 
                      include_top=False, 
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))


# Setting data sets' directories 

TRAIN_DIR = "../final data/train"
VAL_DIR = "../final data/validation"
TEST_DIR = "../final data/test"

# Data Set Translation and Training data scaling

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5,
                    vertical_flip=True
                )
image_gen_val = ImageDataGenerator(rescale=1./255) # translate to RGB
image_gen_test = ImageDataGenerator(rescale=1./255)


train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=TRAIN_DIR,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=VAL_DIR,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')
test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=TEST_DIR,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')



# Set-up FINETUNE MODEL

class_list = ['ind1A', 'ind1B', 'ind2', 'ind3', 'indx']

FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=len(class_list))

NUM_EPOCHS = 500
BATCH_SIZE = 128
num_train_images = 5544 #1000

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])



# TRAINING

# Declaring filepath for checkpoints

filepath = "../checkpoints/" + "inceptionV3" + "_40000.h5" # "_model_weights_final.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

finetune_model.load_weights('../checkpoints/inceptionV3_final_40000.h5') # optional if there are saved model

history = finetune_model.fit_generator(train_data_gen, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True,
                                       callbacks=callbacks_list,
                                       validation_data=val_data_gen)
# save history
with open('../history/train_history_40000', 'wb') as f:
    pickle.dump(history.history, f)


def print_history(max_epoch):
	epoch = 500
	while epoch < max_epoch:
	    print('EPOCH: ' + str(epoch))
	    
	    saved_file = '../history/train_history_' + str(epoch)
	    with open(saved_file, 'rb') as f:
	        # load using pickle de-serializer
	        saved_history = pickle.load(f)
	    
	    print('ACCURACY')
	    print(max(saved_history['accuracy']))
	    print(max(saved_history['val_accuracy']))


	    print('LOSS')
	    print(min(saved_history['loss']))
	    print(min(saved_history['val_loss']))
	    
	    print('')
	    epoch = epoch + 500



# PREDICTION

# loading weights 
finetune_model.load_weights('../checkpoints/inceptionV3_final_40000.h5')

# prediction
predictions = finetune_model.predict_generator(test_data_gen)

# prints only the highest accuracy
def print_pred_max(test_size):
	labels = test_data_gen.class_indices

	for i in range(test_size):
	    pred = np.argmax(predictions[i]) 
	    
	    for cls in labels:
	        if labels[cls] == pred:
	            print('True value: ', class_list[test_data_gen.classes[i]] )
	            print(cls + ': ' +"{:.2%}".format(predictions[i][pred]))
	    print()

# prints prediction with accuracies per class
def print_pred(test_size):
	labels = test_data_gen.class_indices

	for i in range(test_size):
	   
	    print('True value: ', class_list[test_data_gen.classes[i]] )
	    for cls in labels:
	        print(cls + ': ' +"{:.2%}".format(predictions[i][labels[cls]]))
	    print()