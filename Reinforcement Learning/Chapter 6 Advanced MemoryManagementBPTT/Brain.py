import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras
from tensorflow.keras import layers
from IPython import display
import argparse
import wandb
from matplotlib.lines import Line2D
import random
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
#added new
class AntBrain(keras.Model):
    def __init__(self, num_layers, action_network_num_outputs):
        super(AntBrain, self).__init__()
        self.layer1=layers.Dense(num_layers, activation='tanh')
        #self.layer2=layers.Dense(num_layers, activation='tanh')
        self.output_layer=layers.Dense(action_network_num_outputs, activation=None)

    @tf.function
    def call(self, input_vector):
        x=input_vector
        y = self.layer1(x)
        #print(y)
        #y2=self.layer2(y)
        #x=tf.concat([x,y], axis=1)# This adds shortcut connections from the previous layer to the next layer
        #y3=self.layer2(y2)
        #x=tf.concat([x,y], axis=1)# More shortcut connections.
        y4=self.output_layer(y)
        # Using the shortcut connections above means I don't need to worry
        # too much about how many hidden layers to add.  For example, if hidden
        # layers 1 and 2 are not needed then they can simply be skipped over.
        # Also it ensures there are shortcut connections from the input layer to the final layer, which
        # potentially allows memories i
        return y4