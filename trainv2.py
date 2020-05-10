# Downloading dataset
"""
Go to kaggle > account > create API and upload the kaggle.json onto the working directory from the pane on the left. This is required to download datasets using commandline from kaggle
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import argparse

"""# Function to create model"""

def create_model(model_type='max_pool', dropout_rate=0):
    if model_type == 'max_pool':
        img_input = keras.Input(shape=(128,128,1))

        conv1 = keras.layers.Convolution2D(32, (3, 3), activation='relu')(img_input)
        mx_pool = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        drop1 = keras.layers.Dropout(dropout_rate)(mx_pool)

        conv2 = keras.layers.Convolution2D(32, (3, 3), activation='relu')(drop1)
        mx_pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        drop2 = keras.layers.Dropout(dropout_rate)(mx_pool2)

        conv3 = keras.layers.Convolution2D(32, (3, 3), activation='relu')(drop2)
        mx_pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        drop3 = keras.layers.Dropout(dropout_rate)(mx_pool3)

        # Flattening the layers
        flatten1 = keras.layers.Flatten()(drop3)

        # Adding a fully connected layer
        dense1 = keras.layers.Dense(units=128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(flatten1)
        output_layer = keras.layers.Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=img_input, outputs=output_layer)
        return model
    else:
        img_input = keras.Input(shape=(128,128,1))

        conv1 = keras.layers.Convolution2D(32, (3, 3), activation='relu')(img_input)
        avg_pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = keras.layers.Convolution2D(32, (3, 3), activation='relu')(avg_pool1)
        avg_pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = keras.layers.Convolution2D(32, (3, 3), activation='relu')(avg_pool2)
        avg_pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        flatten1 = keras.layers.Flatten()(avg_pool3)

        dense1 = keras.layers.Dense(units=128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(flatten1)
        output_layer = keras.layers.Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=img_input, outputs=output_layer)
        return model


def main(args):
    num_epochs = int(args.num_epochs)
    print("Number of epochs set tp %d" %num_epochs)
    BATCH_SIZE = int(args.batch_size)
    print("Batch_size set to %d" %BATCH_SIZE)
    download_flag = bool(int(args.download))
    if download_flag==True:
        print("Downloading dataset")
        if not os.path.isfile('kaggle.json'):
            print('Get an API token from kaggle!')
            sys.exit()

        os.system('mkdir ~/.kaggle')
        os.system('cp kaggle.json ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')

        """Initiate dataset donwload"""

        os.system('kaggle datasets download -d gti-upm/leapgestrecog')

        """Unzip into data directory"""


        os.system('rm -rf data')
        os.system('mkdir data')
        os.system('unzip -q leapgestrecog.zip -d data')
    else:
        print("Download skipped")

    """More data preprocessing.  
    Note: The data loading function is referenced from [here](https://www.kaggle.com/benenharrington/hand-gesture-recognition-database-with-cnn).
    """

    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('data/leapgestrecog/leapGestRecog/00/'):
        if not j.startswith('.'): # If running this code locally, this is to 
                                # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    lookup

    x_data = []
    y_data = []
    datacount = 0 # We'll use this to tally how many images are in our dataset
    for i in range(0, 10): # Loop over the ten top-level folders
        for j in os.listdir('data/leapgestrecog/leapGestRecog/0' + str(i) + '/'):
            if not j.startswith('.'): # Again avoid hidden folders
                count = 0 # To tally images of a given gesture
                for k in os.listdir('data/leapgestrecog/leapGestRecog/0' + 
                                    str(i) + '/' + j + '/'):
                                    # Loop over the images
                    img = Image.open('data/leapgestrecog/leapGestRecog/0' + 
                                    str(i) + '/' + j + '/' + k).convert('L')
                                    # Read in and convert to greyscale
                    img = img.resize((128, 128))
                    arr = np.array(img)
                    x_data.append(arr) 
                    count = count + 1
                y_values = np.full((count, 1), lookup[j]) 
                y_data.append(y_values)
                datacount = datacount + count
        print("Processing foler %d out of 10" %i)
    x_data = np.array(x_data, dtype = 'float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1) 
    print("Features shape: %s" %str(x_data.shape))
    print("Labels shape: %s" %str(y_data.shape))

    y_onehot = keras.utils.to_categorical(y_data)
    print(y_onehot.shape)

    # Setup training and testing batches

    train_set_x = tf.data.Dataset.from_tensor_slices(x_data[:-1000,:,:])
    train_set_y = tf.data.Dataset.from_tensor_slices(y_onehot[:-1000])
    train_set = tf.data.Dataset.zip((train_set_x, train_set_y)).batch(BATCH_SIZE).shuffle(20000).cache()
    test_set_x = tf.data.Dataset.from_tensor_slices(x_data[-1000:,:,:])
    test_set_y = tf.data.Dataset.from_tensor_slices(y_onehot[-1000:])
    test_set = tf.data.Dataset.zip((test_set_x, test_set_y)).batch(BATCH_SIZE).shuffle(20000).cache()

    model = create_model('avg_pool')
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=1000,
    decay_rate=1,
    staircase=False)
    opt = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Callbacks for early stopping

    es_callback = [
        tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.01,
        patience=5,
        verbose=1,
        mode="auto",
        restore_best_weights=True
    )
    ]
    # "wait for 5 epochs, check if the val_accuracy increases by 0.01, if not, stop and restore the weights that give best val_accuracy"

    # Train model

    model.fit(train_set,epochs=num_epochs,verbose=1,validation_data=test_set,callbacks=es_callback)

    # Save Model

    model_json = model.to_json()
    with open("modelHand.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('modelHand.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and save the model.')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Specify the batch size', default='64')
    parser.add_argument('-e', '--num_epochs', type=int,
                        help='Specify nuber of epochs', default='10')
    parser.add_argument('-d', '--download', type=int,
                        help='Set to 1 for downloading fresh, 0 for not downloading', default='1')
    args = parser.parse_args()
    main(args)