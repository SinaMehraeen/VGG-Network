
# VGG (Visual Geometry Group) network for n class image classification
#ref https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c)
# SM 2022

# Model Architecture 
# series of Conv2D layers followed by a softmax activation layer

import tensorflow as tf
import tensorflow_datasets as tfds
import utils


class Block(tf.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        
        # Define conv2D layers based on n repetitions
        for i in range(0, repetitions):
            
            # Define a Conv2D layer, specifying filters, kernel_size, activation and padding.
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(filters, kernel_size, activation ='relu', padding = 'same')

        # Define the max pool layer that will be added after the Conv2D blocks
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  
    def call(self, inputs):
        # access the class's conv2D_0 layer
        conv2D_0 = vars(self)['conv2D_0']
        
        # Connect the conv2D_0 layer to inputs
        x = conv2D_0(inputs)

        # for the remaining conv2D_i layers from 1 to `repetitions` they will be connected to the previous layer
        for i in range(1, self.repetitions):
            # access conv2D_i by formatting the integer `i`. (hint: check how these were saved using `vars()` earlier)
            conv2D_i = vars(self)[f'conv2D_{i}'] 
            
            # Use the conv2D_i and connect it to the previous layer
            x = conv2D_i(x)

        # Finally, add the max_pool layer
        max_pool = self.max_pool(x)
        
        return max_pool


# ## Create the Custom VGG network 
# This stack has a series of VGG blocks, which can be created using the `Block` class defined earlier.

class MyVGG(tf.keras.Model):

    def __init__(self, num_classes):
        super(MyVGG, self).__init__()

        # Creating blocks of VGG with the following 
        # (filters, kernel_size, repetitions) configurations
        self.block_a = Block(filters = 64, kernel_size = 3, repetitions = 2)
        self.block_b = Block(filters = 128, kernel_size = 3, repetitions = 2)
        self.block_c = Block(filters = 256, kernel_size = 3, repetitions = 3)
        self.block_d = Block(filters = 512, kernel_size = 3, repetitions = 3)
        self.block_e = Block(filters = 512, kernel_size = 3, repetitions = 3)
        

        # Classification head
        # Define a Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.fc = tf.keras.layers.Dense(256, activation = 'relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier = tf.keras.layers.Dense(num_classes, activation = 'softmax')

    def call(self, inputs):
        # Chain all the layers one after the other
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x



# to call model use
# Initialize VGG with the number of classes 
# need to define numnber of classes as int
#myVGGnet = MyVGG(num_classes=n)

# Compile with losses and metrics
#myVGGnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the custom VGG model
#yVGGne.fit(dataset, epochs=10)