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
import progressbar
recordwandb = True
if recordwandb:
    import wandb
from matplotlib.lines import Line2D
from GaussianAntEnvironment import Environment
from Brain import AntBrain
import random
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
#experiment
parser = argparse.ArgumentParser()
parser.add_argument('--sensor_is_directable', type=int, default=0)     #use for network to control the sensors
parser.add_argument('--action_network_sees_xy_position',  type=int, default=0) #add x,y cordinates to the input
parser.add_argument('--sees_highest_points', type=int, default=0)   #add hgihest coordinate to the input
parser.add_argument('--sees_direction_to_highest', type=int, default=0) #compass method
parser.add_argument('--sees_timestep', type = int, default=0)   #add timestep to the input
parser.add_argument('--sensor_distance', type=int, default=0)   #sensor will have a distance from the agent
parser.add_argument('--slow_start', type=int, default=0)    #add random uniform with fixed value at the intialisation of network

parser.add_argument('--n_sensors', type= int, default=1)    #number of sensors used 1 is suggested
parser.add_argument('--num_food', type=int, default=1)  #number of goals in the same map
parser.add_argument('--randomised_height', type = int, default = 1) #randomise the height of the food for each batch
parser.add_argument('--num_memory_nodes', type = int, default= 0)   #number of memory nodes used to input
parser.add_argument('--randomised_food_location', type= int, default=1) #1: agent starts at (0,0) 0: agents scattered around 0,0
parser.add_argument('--memory_mod', type= str, default="No_memory") # Minimal_GRU, CARU, Full_LSTM
parser.add_argument('--randomise_start_trajectory', type= int, default=1) #randomise food or the starting pos at the start of each iteration.
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_layers', type = int, default=20)
parser.add_argument('--trajectory_length', type=int,default=30)
parser.add_argument('--max_iteration', type=int,default=100000)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--printit', type=bool, default=False)
parser.add_argument('--record', type=bool, default=True)
parser.add_argument('--algo_name', type=str, default="BPTT")
parser.add_argument('--trial', type=int, default=3)

args = parser.parse_args()
num_layers = int(args.num_layers)
max_iteration = int(args.max_iteration)
learning_rate = float(args.learning_rate)
randomise_start_trajectory = bool(args.randomise_start_trajectory)
memory_mod = str(args.memory_mod)
num_memory_nodes = int(args.num_memory_nodes)
randomised_food_location = bool(args.randomised_food_location)
sees_timestep = bool(args.sees_timestep)
trajectory_length= int(args.trajectory_length)
randomised_height = bool(args.randomised_height)
num_food = int(args.num_food)
batch_size = int(args.batch_size)
sensor_is_directable = bool(args.sensor_is_directable)
action_network_sees_xy_position = bool(args.action_network_sees_xy_position)
sees_highest_points = bool(args.sees_highest_points)
algo_name = str(args.algo_name)
sees_direction_to_highest = bool(args.sees_direction_to_highest)
slow_start = bool(args.slow_start)
sensor_distance = int(args.sensor_distance)
number_of_sensors = int(args.n_sensors)
render = bool(args.render)
printit =  bool(args.printit)
record = bool(args.record)
trial = int(args.trial)
state_dimension = 2 + (1 if number_of_sensors >0 else 0)
simp_gru = 0
full_gru= 0
full_lstm = 0
no_memory = 0
CARU = 0
original_n_memories = num_memory_nodes
if memory_mod == "Minimal_GRU":
    simp_gru = 1
    num_memory_nodes =  num_memory_nodes * 2  # vx,vy
elif memory_mod == "Full_GRU":
    full_gru =1
    num_memory_nodes =  num_memory_nodes * 3
elif memory_mod == "Full_LSTM":
    full_lstm = 1
    num_memory_nodes = num_memory_nodes *4
elif memory_mod == "CARU":
    CARU = 1
    num_memory_nodes =  num_memory_nodes *3
elif memory_mod == "No_memory":
    num_memory_nodes = 0
    no_memory = 1
else:
    num_memory_nodes =  num_memory_nodes
action_network_num_outputs = 2 + (2 if sensor_is_directable else 0) + num_memory_nodes #x,y,sx,sy,num_mem
action_network_num_inputs = (2 if action_network_sees_xy_position else 0)+number_of_sensors+num_memory_nodes  # x,y
#pointers#
#statedimention 0x,1y,2sensor,memory,3timestep
#action 0velx,1vely,2memory
use_sensor = False
use_memory = False
if number_of_sensors >=1:
    use_sensor = True
if num_memory_nodes >=1:
    use_memory = True
state_memory_pointer = 1 + (1 if use_sensor else 0) + (1 if use_memory else 0)
state_memory_pointer_h = state_memory_pointer
state_memory_pointer_c = state_memory_pointer+ original_n_memories
state_sensor_pointer = 1+ (1 if use_sensor else 0)
action_memory_pointer = 2
state_dimension +=  original_n_memories + (original_n_memories if memory_mod == "Full_LSTM"  else 0) # x,y, timestep, hidden vector memory

print("action_network_num_inputs",action_network_num_inputs )
print("action_network_num_outputs",action_network_num_outputs)
print("state_memory_pointer",state_memory_pointer)
print("state_memory_pointer_h", state_memory_pointer_h)
print("state_memory_pointer_c",state_memory_pointer_c)
print("state_sensor_pointer",state_sensor_pointer)
print("action_memory_pointer",action_memory_pointer)
print("state_dimension",state_dimension)

filename = "Trial_"+ str(trial)+"_Sensors_"+str(number_of_sensors)+"_Memories_"+ str(num_memory_nodes)+"_TypeMemory_"+memory_mod+"_randomisedfood_"+str(randomised_food_location)+"_randomisedheight_"+str(randomised_height)+"_"+"BPTTrunANT"+"_"
if recordwandb:
    wandb.init(name=algo_name, project="memory_modification_experiment", group=algo_name,
               config={
                   "randomised_heights": randomised_height,
                   "memory_modification_type" : memory_mod ,
                   "sensor_distance" : 0,
                   "dimenstional_information" : state_dimension,
                   "number_of_sensors" : number_of_sensors,
                   "num_memory_nodes" : num_memory_nodes,
                   "sensor_is_directable" : sensor_is_directable,
                   "action_network_sees_xy_position" : action_network_sees_xy_position,
                   "randomised_food_location" : randomised_food_location,
                   "sees_highest_points" : sees_highest_points,
                   "sees_direction_to_highest" : sees_direction_to_highest,
                   "neural_layers" :[12,12],
                   "num_food" : num_food,
                   "batch_size" : batch_size,
                   "trajectory_length" : trajectory_length,
                   "max_iteration" : max_iteration,
                   "slow_start" : slow_start,
                   "learning_rate" : learning_rate,
                   "randomise_start_of_iterations": randomise_start_trajectory,
                   "shortcuts": False
               }
               )

fig = plt.figure()
axes = fig.add_subplot(1,1, 1,projection='3d')



environment = Environment(batch_size,randomised_height,randomised_food_location,state_dimension,number_of_sensors,num_memory_nodes,memory_mod)
keras_ant_brain = AntBrain(num_layers,action_network_num_outputs)

def convert_state(state):
    pos_xy = state[:, 0:2]
    sensor = tf.reshape(state[:,state_sensor_pointer], (batch_size,number_of_sensors))
    old_memory_state = state[:, state_memory_pointer:]
    h = state[:,state_memory_pointer_h:state_memory_pointer_c]
    #input_gate, forget_gate, output_gate, h = tf.split(h, num_or_size_splits=4, axis=1)
    c = state[:,state_memory_pointer_c:]
    converted_state = tf.concat([sensor,h], axis=1)
    return converted_state
@tf.function
def expand_full_trajectory(start_states, food_location):
    total_rewards=tf.constant(0.0, dtype=tf.float32, shape=[batch_size])
    state=start_states # this is shape [batch_size, state_dimension]
    trajectory_list=[start_states]
    # build main graph.  This is a long graph with unrolled in time for trajectory_length steps.  Each step includes one neural network followed by one physics-model
    for time_step in range(trajectory_length):
        action = keras_ant_brain(convert_state(state))
        [rewards,state]=environment.run_one_step_of_physics_model(state,action,food_location,state_sensor_pointer,state_memory_pointer,state_memory_pointer_h,state_memory_pointer_c)
        total_rewards+=rewards # This is shape [batch_size]
        trajectory_list.append(state)

    trajectories=tf.stack(trajectory_list) # This will be shape [batch_size, trajectory_length+1, state_dimension]
    average_total_reward=tf.reduce_mean(total_rewards) # this is a scalar
    return [average_total_reward,trajectories]

[average_total_reward,trajectories] = expand_full_trajectory( environment.initial_state, environment.food_location)
fig=None
fig=environment.show_trajectories(trajectories, environment.initial_state, environment.food_location,0, average_total_reward.numpy(), fig)
reward_history=[] # Keep a log for plotting training history
reward_history_validation=[] # Keep a log for plotting training history
reward_history_iters=[] # Keep a log for plotting training history
reward_history_iters_validation=[] # Keep a log for plotting training history

optimizer = keras.optimizers.Adam(learning_rate)

@tf.function
def run_exp(initial_state, food_location):
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        [average_total_reward,trajectories] = expand_full_trajectory(initial_state, food_location)
        loss=-average_total_reward
    grads = tape.gradient(loss,keras_ant_brain.trainable_weights)  # The "back-propagation through time" calculation is the computation of this gradient
    return loss, average_total_reward, trajectories, grads

if printit:
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'),' ', progressbar.Timer(),' ',progressbar.FormatLabel('')]
    bar = progressbar.ProgressBar(maxval=max_iteration,widgets=widgets)
    bar.start()
for i in range(1,max_iteration):
    iteration=len(reward_history)
    if randomise_start_trajectory:
        environment.rerandomise_start()
    loss, average_total_reward , trajectories,grads = run_exp(environment.initial_state, environment.food_location)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, keras_ant_brain.trainable_weights))
    if np.any(np.isnan(trajectories.numpy())):
        print("trajectory",trajectories)
        raise Exception("Trajectories is Nan")
    average_total_reward=average_total_reward.numpy()
    reward_history.append(average_total_reward)
    reward_history_iters.append(iteration)
    if recordwandb:
        wandb.log({"global_step": i,"Training mean Trajectory Reward": average_total_reward})

    if i%100==0:
        [average_total_reward_validation,validation_trajectory] = expand_full_trajectory(environment.initial_state_validation, environment.food_location_validation)
        reward_history_validation.append(average_total_reward_validation)
        reward_history_iters_validation.append(iteration)
        if printit:
            p = "global_step: " + str(i) + " eval/mean_reward: " + str(average_total_reward_validation.numpy())
            widgets[-1] = progressbar.FormatLabel(p.format(i))
            bar.update(i)
            #print()
        if recordwandb:
            wandb.log({"global_step": i, "eval/mean_reward": average_total_reward_validation})
    if render:
        if (len(reward_history)%40)==0:
            fig=environment.show_trajectories(trajectories, environment.initial_state, environment.food_location, len(reward_history), average_total_reward, fig)


if record:
    fig = environment.show_trajectories(trajectories, environment.initial_state, environment.food_location, len(reward_history), average_total_reward, fig)
    plt.savefig(filename+"wandb_upload_train.jpg")
    if recordwandb:
        im = plt.imread(filename+"wandb_upload_train.jpg")
        wandb.log({"train": [wandb.Image(im, caption="Last iteration results")]})
    np.save(filename+"trajectories.npy",validation_trajectory)
    fig = environment.show_trajectories(validation_trajectory, environment.initial_state_validation, environment.food_location_validation, len(reward_history_validation), average_total_reward_validation, fig)
    plt.savefig(filename+"wandb_upload_valid.jpg")
    plt.savefig(filename+"save1.pdf")
    if recordwandb:
        im = plt.imread(filename+"wandb_upload_valid.jpg")
        wandb.log({"validation": [wandb.Image(im, caption="Last iteration results")]})
    fig, ax = plt.subplots()
    ax.set(xlabel='Training Iteration', ylabel='Reward', title='Reward History')
    ax.grid(True)
    ax.plot(reward_history_iters,reward_history,label="train")
    plt.savefig(filename + "train_reward")
    ax.plot(reward_history_iters_validation,reward_history_validation,label="validation")
    fig.legend()
    plt.savefig(filename + "valid_reward")
    np.save(filename+"reward_hisotry_iters",np.array(reward_history_iters))
    np.save(filename+"reward_hisotry",np.array(reward_history))
    np.save(filename+"reward_hisotry_iters_valid",np.array(reward_history_iters_validation))
    np.save(filename+"reward_hisotry_iters_valid",np.array(reward_history_validation))
    if recordwandb:
        wandb.log({"plot":plt})



bar.finish()
