import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import glob2

'''
    Define constant
    - BATCH_SIZE
    - LEARNING_RATE
    - TARGET_SIZE
'''

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

training_directory = './dataset_new/train/'
validation_directory = './dataset_new/validation'

model_name = 'facial_emotion_recognition_new_dataset.h5'

'''
    Initialize data generator
'''
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

'''
    Prepare data iterators
'''
train_iter = train_datagen.flow_from_directory(
    training_directory,
    color_mode='grayscale',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_iter = validation_datagen.flow_from_directory(
    validation_directory,
    color_mode='grayscale',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

'''
    Create custom CNN model
'''
model = Sequential()

### Block 1 initialize
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 input_shape=(48, 48, 1),
                 name='block1_conv1'))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 input_shape=(48, 48, 1),
                 name='block1_conv2'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2),
                       name='block1_maxpool'))
model.add(Dropout(0.2))

### Block 2 initialize
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 name='block2_conv1'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 name='block2_conv2'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2),
                       name='block2_maxpool'))
model.add(Dropout(0.2))

### Block 3 initialize
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 name='block3_conv1'))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 name='block3_conv2'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2),
                       name='block3_maxpool'))
model.add(Dropout(0.2))

### Block 4 initialize
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 name='block4_conv1'))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='elu',
                 padding='same',
                 kernel_initializer='he_normal',
                 name='block4_conv2'))
model.add(BatchNormalization())

### Block 5 initialize
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

### Block 6 initialize
model.add(Dense(len(glob2.glob('dataset_new/train/*')), kernel_initializer='he_normal', activation='softmax'))

### Summarize model information
model.summary()

'''
    Preparing callback function
'''
checkpoint = ModelCheckpoint(
    model_name,
    monitor='val_loss',
    verbose=1,
    mode='auto',
    save_best_only=True,
)

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,  # If the accuracy or less is better, it means the model is better
    patience=3,  # Stop training after 3 epochs if model is overfit
    verbose=1,
    restore_best_weights=True
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_delta=1e-4
)

callbacks = [checkpoint, early_stop, reduce_learning_rate]

'''
    Compile model and start training
'''

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

n_training_sample = 24282
n_validation_sample = 5937
epochs = 25

r = model.fit_generator(
    train_iter,
    steps_per_epoch=n_training_sample//BATCH_SIZE,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_iter,
    validation_steps=n_validation_sample//BATCH_SIZE
)

model.save(model_name)

'''
    Plot accuracy and loss
'''

plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
plt.show()
