# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:27:13 2024

@author: Dhrumit Patel
"""

"""
Get helper functions
"""
# Import series of helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_historys

"""
Use TensorFlow Datasets(TFDS) to download data
"""
# Get TensorFlow Datasets
import tensorflow_datasets as tfds

# List all the available datasets
datasets_list = tfds.list_builders() # Get all available datasets in TFDS
print("food101" in datasets_list) # Is our target dataset in the list of TFDS datasets?

# Load in the data
(train_data, test_data), ds_info = tfds.load(name="food101", 
                                             split=["train", "validation"], 
                                             shuffle_files=True, # Data gets returned in tuple format (data, label)
                                             with_info=True)
# Features of Food101 from TFDS
ds_info.features

# Get the class names
class_names = ds_info.features["label"].names
class_names[:10]

# Take one sample of the train data
train_one_sample = train_data.take(1) # samples are in format (image_tensor, label)
# What does one sample of our training data look like?
train_one_sample

# Output info about our training samples
for sample in train_one_sample:
    image, label = sample["image"], sample["label"]
    print(f"""
    Image shape: {image.shape}
    Image datatype: {image.dtype}
    Target class from Food101 (tensor form): {label}
    Class name (str form): {class_names[label.numpy()]}
    """)

# What does our image tensor from TFDS's Food101 look like?
import tensorflow as tf
image
tf.reduce_min(image), tf.reduce_max(image)

"""
Plot an image from TensorFlow Datasets
"""
# Plot an image tensor
import matplotlib.pyplot as plt
plt.imshow(image)
plt.title(class_names[label.numpy()]) # Add title to verify the label is associated to right image
plt.axis(False)

(image, label)

# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from uint8 -> float32 and reshapes
    image to [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape target image
    # image = image/255. # scale image values (not required for EfficientNet models from tf.keras.applications)
    return tf.cast(image, dtype=tf.float32), label # return a tuple of float32 image and a label tuple

# Preprocess a single sample image and check the outputs
preprocessed_img = preprocess_img(image, label)[0]
print(f"Image before preprocessing:\n {image[:2]}..., \n Shape: {image.shape},\nDatatype: {image.dtype}\n")
print(f"Image after preprocessing:]n {preprocessed_img[:2]}..., \n Shape: {preprocessed_img.shape}, \nDatatype: {preprocessed_img.dtype}")

"""
Batch and preprare datasets

We are now going to make our data input pipeline run really fast.
"""
# Map preprocessing function to training data (and parallelize)
train_data = train_data.map(map_func=lambda sample: preprocess_img(sample['image'], sample['label']), num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turned it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map preprocessing function to test data
test_data = test_data.map(map_func=lambda sample: preprocess_img(sample['image'], sample['label']), num_parallel_calls=tf.data.AUTOTUNE)
# Turn the test data into batches (don't need to shuffle the test data)
test_data = test_data.batch(batch_size=32).prefetch(tf.data.AUTOTUNE)

train_data, test_data

"""
Create modelling callbacks

We are going to create a couple of callbacks to help us while our model trains:
1. TensorBoard callback to log training results (so we can visualize them later if need be)
2. ModelCheckpoint callback to save our model's progress after feature extraction.
"""
# Create tensorboard callback (import from helper_functions.py)
from helper_functions import create_tensorboard_callback

# Create a ModelCheckpoint callback to save a model's progress during training
checkpoint_path = "model_checkpoints/cp.ckpt"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor="val_acc",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      verbose=1)

# Turn on mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16") # Set global data policy to mixed precision
mixed_precision.global_policy()

"""
Build feature extraction model
"""
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Create base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False

# Create functional model
inputs = layers.Input(shape=input_shape, name="input_layer")
# Note: EfficientNetV2B0 models have rescaling built-in but if your model doesn't you can have a layer like below
# x = preprocessing.Rescaling(1./255)(x) 
x = base_model(inputs, training=False) # make sure layers which should be in inference mode only
x = layers.GlobalAveragePooling2D(name="global_pooling_layer")(x)
outputs = layers.Dense(len(class_names), activation="softmax", dtype=tf.float32, name="softmax_float32")(x) # This will be converted to float32

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", # The labels are in integer form
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()

# Check the dtype_policy attributes of layers in our model
for layer in model.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# Check the dtype_policy attributes for the base_model layer
for layer in model.layers[1].layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# OR

for layer in base_model.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# Fit the feature extraction model with callbacks
history_101_food_classes_feature_extract = model.fit(train_data,
                    epochs=3,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=int(0.15 * len(test_data)),
                    callbacks=[create_tensorboard_callback(dir_name="training_logs", experiment_name="efficientnetb0_101_classes_all_data_feature_extract"), model_checkpoint])


# Evaluate the model on the whole test data
results_feature_extract_model = model.evaluate(test_data)
results_feature_extract_model


# 1. Create a function to recreate the original model
def create_model():
  # Create base model
  input_shape = (224, 224, 3)
  base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
  base_model.trainable = False # freeze base model layers

  # Create Functional model 
  inputs = layers.Input(shape=input_shape, name="input_layer")
  # Note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
  # x = layers.Rescaling(1./255)(x)
  x = base_model(inputs, training=False) # set base_model to inference mode only
  x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
  x = layers.Dense(len(class_names))(x) # want one output neuron per class 
  # Separate activation of output layer so we can output float32 activations
  outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) 
  model = tf.keras.Model(inputs, outputs)
  
  return model

# 2. Create and compile a new version of the original model (new weights)
created_model = create_model()
created_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])

# 3. Load the saved weights
created_model.load_weights(checkpoint_path)

# 4. Evaluate the model with loaded weights
results_created_model_with_loaded_weights = created_model.evaluate(test_data)

# 5. Loaded checkpoint weights should return very similar results to checkpoint weights prior to saving
import numpy as np
assert np.isclose(results_feature_extract_model, results_created_model_with_loaded_weights).all(), "Loaded weights results are not close to original model."  # check if all elements in array are close

# Check the layers in the base model and see what dtype policy they're using
for layer in created_model.layers[1].layers[:20]: # check only the first 20 layers to save printing space
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# Save model locally (if you're using Google Colab, your saved model will Colab instance terminates)
save_dir = "07_efficientnetb0_feature_extract_model_mixed_precision"
model.save(save_dir)

# Load model previously saved above
loaded_saved_model = tf.keras.models.load_model(save_dir)

# Load model previously saved above
loaded_saved_model = tf.keras.models.load_model(save_dir)


# Check the layers in the base model and see what dtype policy they're using
for layer in loaded_saved_model.layers[1].layers[:20]: # check only the first 20 layers to save output space
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

results_loaded_saved_model = loaded_saved_model.evaluate(test_data)
results_loaded_saved_model

# The loaded model's results should equal (or at least be very close) to the model's results prior to saving
import numpy as np
assert np.isclose(results_feature_extract_model, results_loaded_saved_model).all()


"""
Optional
"""
# Download and unzip the saved model from Google Storage - https://drive.google.com/file/d/1-4BsHQyo3NIBGzlgqZgJNC5_3eIGcbVb/view?usp=sharing

# Unzip the SavedModel downloaded from Google Storage
# !mkdir downloaded_gs_model # create new dir to store downloaded feature extraction model
# !unzip 07_efficientnetb0_feature_extract_model_mixed_precision.zip -d downloaded_gs_model

# Load and evaluate downloaded GS model
loaded_gs_model = tf.keras.models.load_model("downloaded_gs_model/07_efficientnetb0_feature_extract_model_mixed_precision")

# Get a summary of our downloaded model
loaded_gs_model.summary()

# How does the loaded model perform?
results_loaded_gs_model = loaded_gs_model.evaluate(test_data)
results_loaded_gs_model

# Are any of the layers in our model frozen?
for layer in loaded_gs_model.layers:
    layer.trainable = True # set all layers to trainable
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")


# Check the layers in the base model and see what dtype policy they're using
for layer in loaded_gs_model.layers[1].layers[:20]:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training

# Create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = "fine_tune_checkpoints/"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")
# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

# Compile the model
loaded_gs_model.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                        optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                        metrics=["accuracy"])


# Start to fine-tune (all layers)
history_101_food_classes_all_data_fine_tune = loaded_gs_model.fit(train_data,
                                                        epochs=100, # fine-tune for a maximum of 100 epochs
                                                        steps_per_epoch=len(train_data),
                                                        validation_data=test_data,
                                                        validation_steps=int(0.15 * len(test_data)), # validation during training on 15% of test data
                                                        callbacks=[create_tensorboard_callback("training_logs", "efficientb0_101_classes_all_data_fine_tuning"), # track the model training logs
                                                                   model_checkpoint, # save only the best model during training
                                                                   early_stopping, # stop model after X epochs of no improvements
                                                                   reduce_lr]) # reduce the learning rate after X epochs of no improvements

# Save model locally (note: if you're using Google Colab and you save your model locally, it will be deleted when your Google Colab session ends)
loaded_gs_model.save("07_efficientnetb0_fine_tuned_101_classes_mixed_precision")


"""
Optional
"""
# Download and evaluate fine-tuned model from Google Storage - https://drive.google.com/file/d/1owx3maxBae1P2I2yQHd-ru_4M7RyoGpB/view?usp=sharing

# Unzip fine-tuned model
# !mkdir downloaded_fine_tuned_gs_model # create separate directory for fine-tuned model downloaded from Google Storage
# !unzip 07_efficientnetb0_fine_tuned_101_classes_mixed_precision -d downloaded_fine_tuned_gs_model

# Load in fine-tuned model and evaluate
loaded_fine_tuned_gs_model = tf.keras.models.load_model("downloaded_fine_tuned_gs_model/07_efficientnetb0_fine_tuned_101_classes_mixed_precision")

# Get a model summary 
loaded_fine_tuned_gs_model.summary()

# Note: Even if you're loading in the model from Google Storage, you will still need to load the test_data variable for this cell to work
results_downloaded_fine_tuned_gs_model = loaded_fine_tuned_gs_model.evaluate(test_data)
results_downloaded_fine_tuned_gs_model

"""
# Upload experiment results to TensorBoard (uncomment to run)
# !tensorboard dev upload --logdir ./training_logs \
#   --name "Fine-tuning EfficientNetB0 on all Food101 Data" \
#   --description "Training results for fine-tuning EfficientNetB0 on Food101 Data with learning rate 0.0001" \
#   --one_shot

# View past TensorBoard experiments
# !tensorboard dev list


# Delete past TensorBoard experiments
# !tensorboard dev delete --experiment_id YOUR_EXPERIMENT_ID

# Example
# !tensorboard dev delete --experiment_id OAE6KXizQZKQxDiqI3cnUQ
"""

















