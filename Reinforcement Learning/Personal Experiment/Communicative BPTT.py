import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from matplotlib import pyplot as plt
class network(keras.Model):
    def __init__(self, num_layers, action_network_num_outputs):
        super(network, self).__init__()
        self.layer1 = layers.Dense(num_layers, activation='tanh',dtype=tf.float64)
        # self.layer2=layers.Dense(num_layers, activation='tanh')
        self.output_layer = layers.Dense(action_network_num_outputs, activation=None,dtype=tf.float64)

    @tf.function
    def call(self, input_vector):
        x= input_vector
        y = self.layer1(x)
        out = self.output_layer(y)
        return out


class Differentiable_Cartpole():

    def __init__(self, batch_size = 32):
        super(Differentiable_Cartpole).__init__()
        self.batch_size = batch_size
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step duration
        self.initial_state = self._randomize_state()
    def _randomize_state(self):

        state = tf.concat([np.random.uniform(-0.05, 0.05,size=(self.batch_size, 1)),
                          np.random.uniform(-0.05, 0.05, size=(self.batch_size, 1)),
                          np.random.uniform(-0.05, 0.05, size=(self.batch_size, 1)),
                          np.random.uniform(-0.05, 0.05, size=(self.batch_size, 1))],axis=1
                          )
        return state
    def reset(self):

        self.initial_state = self._randomize_state()

    def step(self,state, action):
        x, x_dot, theta, theta_dot = tf.unstack(state, axis = 1)

        force = tf.reshape(action * self.force_mag, (batch_size,))


        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)
        # For the pole
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass

        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))

        # For the cart
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        # Update the state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc

        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        reward = tf.abs(tf.tanh(theta))
        x = tf.reshape(x, (batch_size,1))
        x_dot = tf.reshape(x_dot, (batch_size, 1))
        theta = tf.reshape(theta, (batch_size, 1))
        theta_dot = tf.reshape(theta_dot, (batch_size, 1))
        state = tf.concat([x, x_dot, theta, theta_dot], axis=1)

        return [reward, state]

    def render(self, states,reward):

        # states should have shape [number of steps, batch size, 4]
        avg_states = tf.reduce_mean(states, axis=1).numpy()
        num_steps = avg_states.shape[0]
        time = np.arange(num_steps) * self.tau
        # Plot cart position
        axs[0].lines.clear()
        axs[0].plot(time, avg_states[:, 0], label='Cart Position',color="b")
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position (m)')
        axs[0].set_title('Cart Position over Time')
        axs[0].legend()

        # Plot pole angle
        axs[1].lines.clear()
        axs[1].plot(time, avg_states[:, 2], label='Pole Angle',color="b")
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Angle (rad)')
        axs[1].set_title('Pole Angle over Time')
        axs[1].legend()

        # Plot cart velocity
        axs[2].lines.clear()
        axs[2].plot(time, avg_states[:, 1], label='Cart Velocity',color="b")
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Velocity (m/s)')
        axs[2].set_title('Cart Velocity over Time')
        axs[2].legend()
        axs[3].lines.clear()
        axs[3].plot( np.arange(len(reward)),reward,label='Average Reward', color="b")
        axs[3].set_xlabel('Iteration')
        axs[3].set_ylabel('Average Reward Theta Angle of the Pole')
        axs[3].set_title('Reward over Iteration')
        axs[3].legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

def convert_state(state):
    x, x_dot, theta, theta_dot = tf.unstack(state, axis=1)
    x= tf.reshape(tf.tanh(x), (batch_size, 1))
    x_dot = tf.reshape(tf.tanh(x_dot), (batch_size, 1))
    theta = tf.reshape(tf.tanh(theta), (batch_size, 1))
    theta_dot = tf.reshape(tf.tanh(theta_dot), (batch_size, 1))
    state= tf.concat([x,x_dot,theta,theta_dot], axis=1)
    return state



fig, axs = plt.subplots(4, 1, figsize=(15, 10))
batch_size=  10
environment = Differentiable_Cartpole(batch_size)
brain = network(20,1)
trajectory_length = 500
optimizer = keras.optimizers.Adam()
@tf.function
def expand_full_trajectory(start_states):

    total_rewards = tf.constant(0.0, dtype= tf.float64, shape= [batch_size])
    state = start_states
    trajectory_list = [start_states]
    for t in range(trajectory_length):

        action = brain(convert_state(state))
        [rewards, state] = environment.step(state, action)
        total_rewards += rewards
        trajectory_list.append(state)

    trajectories = tf.stack(trajectory_list)
    average_total_rewards = tf.reduce_mean(total_rewards)
    return [average_total_rewards,trajectories]

@tf.function
def run_exp(initial_state):
    with tf.GradientTape() as tape:
        [average_total_reward, trajectories] = expand_full_trajectory(initial_state)
        loss = average_total_reward
    grads = tape.gradient(loss,brain.trainable_weights)
    return loss, average_total_reward, trajectories, grads

avg_hist = []
for i in range(1, 10000):
    environment.reset()
    loss, average_total_reward , trajectories,grads = run_exp(environment.initial_state)
    avg_hist.append(average_total_reward)
    optimizer.apply_gradients(zip(grads, brain.trainable_weights))
    environment.render(trajectories,avg_hist)






