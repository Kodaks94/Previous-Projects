import sys

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
class Environment:
    def __init__(self, batch_size, randomise_food_heights,randomise_food_location, state_dimension, use_sensor,num_memory_nodes,type_mem):
        self.randomise_food_location = randomise_food_location
        self.type_mem = type_mem
        self.use_sensor = use_sensor
        self.batch_size = batch_size
        self.randomise_food_heights = randomise_food_heights
        self.num_memory_nodes = num_memory_nodes
        self.food_location = tf.constant( (np.random.rand(batch_size,3)-0.5) *(np.array(
            [8,8,randomise_food_heights]) if randomise_food_location else np.array([0,0,0])), tf.float32)
        self.food_location_validation = tf.constant((np.random.rand(batch_size, 3) - 0.5) * (
            np.array([8, 8, randomise_food_heights]) if randomise_food_location else np.array([0, 0, 0])), tf.float32)
        self.initial_state = tf.concat(
            [tf.constant((np.random.rand(batch_size, 2) - 0.5) * (0 if randomise_food_location else 8), tf.float32),
             tf.zeros([batch_size, state_dimension - 2], tf.float32)], axis=1)
        self.initial_state_validation = tf.concat(
            [tf.constant((np.random.rand(batch_size, 2) - 0.5) * (0 if randomise_food_location else 8), tf.float32),
             tf.zeros([batch_size, state_dimension - 2], tf.float32)], axis=1)

    def food_density(self,xy_position, food_location):
        # Define a gaussian bump:
        bump_width = 8
        result = tf.exp(tf.reduce_sum(-(xy_position - food_location[:, :2]) ** 2, axis=1) / bump_width)
        result = result + food_location[:, 2]
        return result

    def plot_food_pile(self,axes,food_location):
        X = np.linspace(-5, 5)
        Y = np.linspace(-5, 5)
        X, Y = np.meshgrid(X, Y)
        xy_grid = np.stack([X, Y], axis=2).reshape((-1, 2))
        Z = tf.reshape(self.food_density(xy_grid, food_location[0:1, :]), [50, 50])
        axes.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis, alpha=0.3)
        cset = axes.contourf(X, Y, Z, zdir='z', offset=0, colors=['#808080', '#A0A0A0', '#C0C0C0'], alpha=0.4)
        # limits ticks and view angle
        axes.set_zlim(-0.5, 1.2)
        axes.set_zticks(np.linspace(0, 1, 5))
        axes.view_init(27, -21)

    def sensor_calculation(self,pos, food_location):
        sensor_result = self.food_density(pos, food_location)
        sensor_result = tf.reshape(sensor_result, [self.batch_size, 1])  # reshape it to a rank-2 tensor
        return sensor_result
    @tf.function
    def run_one_step_of_physics_model(self,state, action, food_location,state_sensor_pointer,state_memory_pointer,state_memory_pointer_h,state_memory_pointer_c):
        '''
        State
        '''
        pos_xy = state[:, 0:2]
        if self.use_sensor:
            sensor = tf.reshape(state[:, state_sensor_pointer], (self.batch_size, int(self.use_sensor)))
        old_memory_state = state[:, state_memory_pointer:]
        if self.type_mem == "Full_LSTM":
            h = state[:, state_memory_pointer_h:state_memory_pointer_c]
            c = state[:, state_memory_pointer_c:]
        '''
        Action
        '''
        vel_xy = action[:, 0:2]
        # Note this assumes there is no tanh on the final layer!
        vel_xy_magnitude = tf.sqrt(tf.reduce_sum(tf.square(vel_xy), axis=1))
        vel_xy_magnitude = tf.expand_dims(vel_xy_magnitude, 1)
        vel_xy_normalised = vel_xy / (vel_xy_magnitude + 1e-6)
        vel_xy = vel_xy_normalised * tf.tanh(vel_xy_magnitude)
        next_pos_xy = pos_xy + vel_xy * 0.2
        next_state_list = [next_pos_xy]
        # Note for sensor functionality we have the sensor_calculation function above to call upon.
        # use next_state_list.append(...) to add new chunks of the tensor you are building up
        if self.use_sensor:
            next_state_list.append(self.sensor_calculation(next_pos_xy,food_location))
        if self.num_memory_nodes > 0:
            action_memory = action[:, 2:]

            if self.type_mem == "Minimal_GRU":
                # A simplified version of the GRU
                input_gate, new_value = tf.split(value=action_memory,
                                                 num_or_size_splits=2, axis=1)
                # output_gate=-input_gate
                # forget_bias_tensor = 1.0
                # hidden_c = ((hidden_c * tf.sigmoid(forget_gate+forget_bias_tensor)) +
                #         (tf.sigmoid(input_gate) * tf.tanh(new_value)))
                #   action_memory = (old_memory_state * (1 - input_gate)) + (input_gate * tf.tanh(new_value))
                input_gate = tf.sigmoid(input_gate)
                action_memory = (tf.tanh(new_value)*input_gate + old_memory_state * (1 - input_gate))

            elif self.type_mem == "Full_GRU":
                #matrix_x, matrix_inner = tf.split(value=action_memory, num_or_size_splits=2, axis=1)
                #x_z, x_r, x_h = tf.split(matrix_x,3,axis=1)
                #recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, 3, axis=1)
                #z = tf.sigmoid(x_z + recurrent_z)
                #r = tf.sigmoid(x_r + recurrent_r)
                #hh = tf.tanh(x_h + r * recurrent_h)
                #action_memory = z *tf.tanh(old_memory_state)+ hh * (1 - z)
                x_z,x_r,x_h = tf.split(value=action_memory, num_or_size_splits=3, axis=1)
                z = tf.sigmoid(x_z)
                r = tf.sigmoid(x_r)
                hh = tf.tanh(x_h*r)
                action_memory = (1-z) *old_memory_state + z*hh
            elif self.type_mem == "Full_LSTM":
                #state_vector  actionmemory,c - > action memory is shown to neural network
                input_gate,forget_gate,output_gate,new_value = tf.split(action_memory,num_or_size_splits=4,axis=1)
                i = tf.sigmoid(input_gate)
                f = tf.sigmoid(forget_gate)
                c = f * c + i * tf.tanh(new_value)
                o = tf.sigmoid(output_gate)
                h = o * tf.tanh(c)
                action_memory = tf.concat([h,c],axis=1)
            elif self.type_mem == "CARU":
                x , a, b = tf.split(action_memory,num_or_size_splits=3,axis=1)
                n = tf.tanh(a)
                l = tf.sigmoid(x)*tf.sigmoid(b)
                action_memory = (1-l)*old_memory_state + l *n

            elif self.type_mem == "No_memory":
                pass

            else:
                action_memory = tf.tanh(action_memory)
            next_state_list.append(action_memory)
        # END of code block in which to insert lines for CHALLENGE 2
        next_state = tf.concat(next_state_list,
                               axis=1)  # appends the rank-2 tensors in next_state_list side-by-side into one rank-2 tensor

        rewards = self.food_density(next_state[:, 0:2], food_location)
        return [rewards, next_state]

    def add_arrow_to_line2D( self,axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>', arrowsize=1, transform=None):
        if not isinstance(line, mlines.Line2D):
            raise ValueError("expected a matplotlib.lines.Line2D object")
        x, y = line.get_xdata(), line.get_ydata()
        arrow_kw = {
            "arrowstyle": arrowstyle,
            "mutation_scale": 10 * arrowsize,
        }
        color = line.get_color()
        use_multicolor_lines = isinstance(color, np.ndarray)
        if use_multicolor_lines:
            raise NotImplementedError("multicolor lines not supported")
        else:
            arrow_kw['color'] = color
        linewidth = line.get_linewidth()
        if isinstance(linewidth, np.ndarray):
            raise NotImplementedError("multiwidth lines not supported")
        else:
            arrow_kw['linewidth'] = linewidth
        if transform is None:
            transform = axes.transData
        arrows = []
        for loc in arrow_locs:
            s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
            n = np.searchsorted(s, s[-1] * loc)
            arrow_tail = (x[n], y[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
            p = mpatches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform,
                **arrow_kw)
            axes.add_patch(p)
            arrows.append(p)
        return arrows

    def show_trajectories(self,trajectories, initial_state, food_location, iteration_number, reward, fig0):
        bs = self.batch_size
        reward = np.mean(reward)
        if trajectories.shape[1] > 10:
            bs = 10
        if fig0 != None:
            plt.close(fig0)
        display.clear_output(wait=True)
        fig = plt.figure(figsize=[5, 5])
        axes_2d = fig.add_subplot(1, 1, 1)
        axes_2d.axis([-6, 6, -6, 6])
        colcycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                    '#17becf']
        for traj in range(bs):
            axes_2d.scatter(food_location[traj, 0], food_location[traj, 1], marker="x",
                            c=colcycle[traj % len(colcycle)])
            axes_2d.scatter(initial_state[traj, 0], initial_state[traj, 1], marker="o",
                            c=colcycle[traj % len(colcycle)])
        for traj in range(bs):
            trajectory_x = trajectories[:, traj, 0]
            trajectory_y = trajectories[:, traj, 1]
            lines, = axes_2d.plot(trajectory_x, trajectory_y, '-', label='Traj1', c=colcycle[traj % len(colcycle)])

            self.add_arrow_to_line2D(axes_2d, lines, arrow_locs=np.array([0.5
                                                                     ]),
                                arrowstyle='->', arrowsize=1.2)
        axes_2d.grid(True)
        if self.randomise_food_location == 0:
            axes_2d.set_title('Top-Down view. Food pile fixed at (0,0).')
        else:
            axes_2d.set_title('Top-Down view.  Agent starting location fixed at (0,0).')
        '''
        if not(randomise_food_location):
            # since there is only one food location, we can do a 3d plot too
            axes_3d=fig.add_subplot(1,2, 2,projection='3d')
            plot_food_pile(axes_3d)
            for traj in range(bs):
                trajectory_x = trajectories[:, traj, 0]
                trajectory_y = trajectories[:, traj, 1]
                tZ=food_density(trajectories[:, traj, 0:2],food_location[traj:traj+1, :])
                axes_3d.plot(trajectory_x, trajectory_x,tZ , c=colcycle[traj%len(colcycle)])

            axes_3d.set_title('3d view')
        '''
        if fig0 != None:
            display.display(plt.gcf())
        return fig
    def rerandomise_start(self):

        self.food_location = tf.constant( (np.random.rand(self.batch_size,3)-0.5) *(np.array([8,8,self.randomise_food_heights]) if self.randomise_food_location else np.array([0,0,0])), tf.float32)


