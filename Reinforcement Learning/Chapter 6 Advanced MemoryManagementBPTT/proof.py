import numpy as np

from GaussianAntEnvironment import Environment
import tensorflow as tf
number_of_sensors = 1
num_memory_nodes = 0
memory_mod = ""
sensor_is_directable = False
action_network_sees_xy_position= False

state_dimension = 2 + (1 if number_of_sensors >0 else 0)
simp_gru = 0
full_gru= 0
full_lstm = 0
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
randomised_height = True
randomised_food_location = True

env = Environment(10**6,randomised_height,randomised_food_location,state_dimension,number_of_sensors,num_memory_nodes,memory_mod)
action = tf.constant( np.zeros([10**6,action_network_num_outputs]),tf.float32)
[reward,state] = env.run_one_step_of_physics_model(env.initial_state,action,env.food_location,state_sensor_pointer,state_memory_pointer,state_memory_pointer_h,state_memory_pointer_c)
reward = tf.reduce_mean(reward)
print(reward*30)
