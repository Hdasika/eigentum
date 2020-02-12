import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import keras
#config = tf.ConfigProto( device_count = {'GPU': 1 } ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
#import keras as K1
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_set = train_datagen.flow_from_directory('./train', target_size=(64, 64), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True, seed=42)

valid_datagen = ImageDataGenerator(rescale=1./255)

valid_set = valid_datagen.flow_from_directory('./val', target_size=(64, 64), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True, seed=42)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory( './test', target_size=(64, 64), color_mode="rgb", batch_size=1, class_mode=None, shuffle=False, seed=42)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten
model = Sequential()
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#get number of columns in training data
#n_cols = train_X.shape[1]

#add model layers
model.add(Conv2D(40, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(Conv2D(70, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(82, activation='softmax'))
model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu', input_shape=(224,224,3)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(100))

#train_generator
model.compile(optimizer='adam', loss='mean_squared_error')
STEP_SIZE_TRAIN=train_set.n//train_set.batch_size
STEP_SIZE_VALID=valid_set.n//valid_set.batch_size

model.fit_generator(generator=train_set,steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_set, validation_steps=STEP_SIZE_VALID, epochs=1)

model.evaluate_generator(generator=valid_set)
model.save('model_1.h5')

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

#model.save('model_1.h5')

'''import sys 
from PIL import Image 
import tensorflow as tf'''

'''def convert_to_tfrecord(dataset_name, data_directory, class_map, segments=1, directories_as_labels=True, files='**/*.jpg'):

    # Create a dataset of file path and class tuples for each file
    filenames = glob.glob(os.path.join(data_directory, files))
    classes = (os.path.basename(os.path.dirname(name)) for name in filenames) if directories_as_labels else [None] * len(filenames)
    dataset = list(zip(filenames, classes))

    # If sharding the dataset, find how many records per file
    num_examples = len(filenames)
    samples_per_segment = num_examples // segments

    print(f"Have {samples_per_segment} per record file")

    for segment_index in range(segments):
        start_index = segment_index * samples_per_segment
        end_index = (segment_index + 1) * samples_per_segment

        sub_dataset = dataset[start_index:end_index]
        record_filename = os.path.join(data_directory, f"{dataset_name}-{segment_index}.tfrecords")

        with tf.python_io.TFRecordWriter(record_filename) as writer:
            print(f"Writing {record_filename}")

            for index, sample in enumerate(sub_dataset):
                sys.stdout.write(f"\rProcessing sample {start_index+index+1} of {num_examples}")
                sys.stdout.flush()

                file_path, label = sample
                image = Image.open(file_path)
                image = image.resize((224, 224))
                image_raw = np.array(image).tostring()

                features = {
                    'label': _int64_feature(class_map[label]),
                    'text_label': _bytes_feature(label),
                    'image': _bytes_feature(image_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())'''

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.tree import DecisionTreeRegressor

# print(__doc__)

# # construct the dataset
# rnd = np.random.RandomState(42)
# X = rnd.uniform(-3, 3, size=100)
# y = np.sin(X) + rnd.normal(size=len(X)) / 3
# X = X.reshape(-1, 1)

# # transform the dataset with KBinsDiscretizer
# enc = KBinsDiscretizer(n_bins=10, encode='onehot')
# X_binned = enc.fit_transform(X)

# # predict with original dataset
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
# line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
# reg = LinearRegression().fit(X, y)
# ax1.plot(line, reg.predict(line), linewidth=2, color='green',
#          label="linear regression")
# reg = DecisionTreeRegressor(min_samples_split=3, random_state=0).fit(X, y)
# ax1.plot(line, reg.predict(line), linewidth=2, color='red',
#          label="decision tree")
# ax1.plot(X[:, 0], y, 'o', c='k')
# ax1.legend(loc="best")
# ax1.set_ylabel("Regression output")
# ax1.set_xlabel("Input feature")
# ax1.set_title("Result before discretization")

# # predict with transformed dataset
# line_binned = enc.transform(line)
# reg = LinearRegression().fit(X_binned, y)
# ax2.plot(line, reg.predict(line_binned), linewidth=2, color='green',
#          linestyle='-', label='linear regression')
# reg = DecisionTreeRegressor(min_samples_split=3,
#                             random_state=0).fit(X_binned, y)
# ax2.plot(line, reg.predict(line_binned), linewidth=2, color='red',
#          linestyle=':', label='decision tree')
# ax2.plot(X[:, 0], y, 'o', c='k')
# ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1, alpha=.2)
# ax2.legend(loc="best")
# ax2.set_xlabel("Input feature")
# ax2.set_title("Result after discretization")

# plt.tight_layout()
# plt.show()

#!nvidia-smi

