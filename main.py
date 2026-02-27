# credit: https://www.geeksforgeeks.org/computer-vision/deep-dream-an-in-depth-exploration/ 

import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
tf.__version__ 

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
#base_model.summary() 

names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]
deep_dream_model = tf.keras.Model(inputs = base_model.input, outputs = layers)

image = tf.keras.preprocessing.image.load_img('./content/neighborhood.jpg',
                                              target_size=(225, 375))
plt.imshow(image)
image = tf.keras.preprocessing.image.img_to_array(image)
# image = image / 255
image = tf.keras.applications.inception_v3.preprocess_input(image)

image_batch = tf.expand_dims(image, axis = 0)

activations = deep_dream_model.predict(image_batch)

activations[0].shape, activations[1].shape

def calculate_loss(image, network):
  image_batch = tf.expand_dims(image, axis = 0)
  activations = network(image_batch)

  losses = []
  for act in activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  print(losses)
  #print(np.shape(losses))
  #print(tf.reduce_sum(losses))

  return tf.reduce_sum(losses)

loss = calculate_loss(image, deep_dream_model)

# Compare the activations with the pixels
# Emphasize parts of the image
# Change the pixels of the input image

@tf.function
def deep_dream(network, image, learning_rate):
  with tf.GradientTape() as tape:
    tape.watch(image)
    loss = calculate_loss(image, network)

  gradients = tape.gradient(loss, image) # Derivate
  gradients /= tf.math.reduce_std(gradients)
  image = image + gradients * learning_rate
  image = tf.clip_by_value(image, -1, 1)

  return loss, image

def inverse_transform(image):
  image = 255 * (image + 1.0) / 2.0
  return tf.cast(image, tf.uint8)

def run_deep_dream(network, image, epochs, learning_rate):
  for epoch in range(epochs):
    loss, image = deep_dream(network, image, learning_rate)

    if epoch % 9999 == 0:
      plt.figure(figsize=(12,12))
      plt.imshow(inverse_transform(image))
      plt.show()
      print('Epoch {}, loss {}'.format(epoch, loss))

run_deep_dream(network=deep_dream_model, image=image, epochs = 10000, learning_rate=0.01)


