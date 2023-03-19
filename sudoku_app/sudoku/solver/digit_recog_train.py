import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import numpy as np

ASSET_DIR = "/Users/timoh/OneDrive/Dokumente/Bildung/Programmieren/old_version/assets/"

def add_border(np_img, border_width):
    height,width =np_img.shape
    img_with_border = np.zeros((height+2*border_width,width+2*border_width))
    for i in range(height):
        for j in range(width):
            img_with_border[i+border_width,j+border_width]=np_img[i,j]
    return img_with_border
            
def preprocess_vgg(np_image):
    np_image=add_border(np_image,2)
    np_image=np.expand_dims(np_image,axis=-1)
    rgb_img = np.repeat(np_image[..., np.newaxis], 3, -1)
    rgb_img = np.resize(rgb_img,(1,32,32,3))
    return rgb_img


mnist = tf.keras.datasets.mnist

input_shape = (32,32,3)

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0
border_width = 2
x_train = [preprocess_vgg(img) for img in x_train]
x_test = [preprocess_vgg(img) for img in x_test]
#x_train = np.expand_dims(x_train,-1)
#x_test = np.expand_dims(x_test,-1)

vgg = VGG16(weights='imagenet',include_top=False, input_shape=input_shape)

# Freeze all layers in VGG model
for layer in vgg.layers:
    layer.trainable = False

# Add custom output layers to VGG model
x = Flatten()(vgg.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)

# Create new model with VGG base and custom output layers
model = Model(inputs=vgg.input, outputs=x)

# Compile the model
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

"""
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
"""
model.fit(x_train, y_train, batch_size=32,epochs=5,validation_split=0.1)
model.save(ASSET_DIR+"digit_model_2.h5")