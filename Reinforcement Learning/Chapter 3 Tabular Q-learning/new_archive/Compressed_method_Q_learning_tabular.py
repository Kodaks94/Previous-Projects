import random
import sys

import numpy as np


def environment_step(tempmaze,action,state,goal_state):
    y,x = state
    dy,dx=action_effects[action]
    hit_wall = False
    new_x = x+dx
    new_y = y+dy
    if new_x <0 or new_x>=maze_width:
        # off grid
        new_x = x
    if new_y <0 or new_y>=maze_height:
        # off grid
        new_y = y
    if tempmaze[new_y,new_x] == 1:
        # hit wall
        new_y=y
        new_x=x
        hit_wall = True
    new_state = [new_y,new_x]

    reward = -1
    done = (new_state==goal_state)
    return new_state, reward, done, hit_wall

def run_policy(currentState, ep,direction_of_entry):
    sy,sx= currentState
    if np.random.uniform() < ep:
        choice = np.random.choice(range(len(action_names)))
    else:
        q_values = Qtable2[sy,sx,:,:]
        q_values = np.array([q_values[visited_counts[sy, sx,direction_of_entry], i] for i in range(num_actions)])
        assert len(q_values.shape)==1
        assert q_values.shape[0]==num_actions
        best_q_value=q_values.max()
        best_q_indices=np.argwhere(q_values == best_q_value).flatten().tolist()
        choice = np.random.choice(best_q_indices)
        assert choice>=0 and choice<num_actions
    return choice


def apply_q_update(state, action, visited_counts,direction_of_entry,next_direction_of_entry,reward, next_state, done):
    sy,sx=state
    nsy,nsx=next_state
    current_q_value = Qtable2[sy, sx, visited_counts[sy, sx,direction_of_entry], action]
    target_q_value = reward
    if not done:
        future_state_q_values = Qtable2[nsy,nsx, :, :]
        future_state_q_values = np.array([future_state_q_values[visited_counts[nsy,nsy,next_direction_of_entry],i] for i in range(num_actions)])
        assert len(future_state_q_values.shape)==1
        assert future_state_q_values.shape[0]==num_actions
        target_q_value += discount_factor * future_state_q_values.max()
        #Qtable2[sy, sx, visited_counts[sy, sx],action] += learning_rate * (target_q_value - current_q_value)  # update
    Qtable2[sy, sx, visited_counts[sy,sx,direction_of_entry],action] += learning_rate * (target_q_value - current_q_value)  # update
    return state

action_names=["North","South","West","East"]
action_effects=[[-1,0],[1,0],[0,-1],[0,1]]
num_actions=len(action_names)
iterations = 500 *4 * 10
#iterations = 5000
learning_rate = 0.1
epsilon_greedy = 0.1
discount_factor=0.9
maps = []
'''
name = 'smallCorridor'
SmallCorridor=np.array([
        [0,0,0],])
SmallCorridor_goal_state = [[0,0],[0,2]]
SmallCorridor_start_state = [0,1]
maps.append([name,SmallCorridor,SmallCorridor_start_state,SmallCorridor_goal_state,3])
'''
name = 'smallCorridor'
SmallCorridor=np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,0,0,0,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1]])
SmallCorridor_goal_state = [[3,2],[3,4]]
SmallCorridor_start_state = [3,3]
maps.append([name,SmallCorridor,SmallCorridor_start_state,SmallCorridor_goal_state,3])

#
name = "Tshaped"
Tshaped =np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1]])
Tshaped_goal_states = [[5,1],[5,5]]
#Tshaped_goal_states = [[5,1]]
Tshaped_start_state = [1,3]
maps.append([name,Tshaped,Tshaped_start_state,Tshaped_goal_states,10])
#
name = 'LongCorridor'
LongCorridor=np.array([
        [1,1,1,1,1,1,1],
        [1,0,1,0,0,0,1],
        [1,0,1,1,1,0,1],
        [1,0,1,0,0,0,1],
        [1,0,1,1,1,0,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1]])
LongCorridor_goal_state = [[1,3],[3,3]]
LongCorridor_start_state = [1,1]
maps.append([name,LongCorridor,LongCorridor_start_state,LongCorridor_goal_state,18])
#
name = 'crossmap'
cross=np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1]])
cross_start_state = [3,3] # center square
cross_goal_states = [[1,3],[3,5],[5,3],[3,1]]
maps.append([name,cross,cross_start_state,cross_goal_states,12])
'''
#
'''
name = 'complex_looped'
complex_looped = np.array([
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1],
[1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1],
[1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1],
[1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1],
[1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,1],
[1,0,0,0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,1],
[1,0,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1],
[1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1],
[1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
[1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1],
[1,0,1,1,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1],
[1,0,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,1,0,1],
[1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1],
[1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1],
[1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])
height= len(complex_looped)
width = len(complex_looped[0])
#complex_looped = np.genfromtxt("foo.csv",delimiter=',')
complexlooped_start_state = [10,10]
complexlooped_goal_states= [[1,1],[1,18],[18,1],[18,18]]
maps.append([name,complex_looped,complexlooped_start_state,complexlooped_goal_states,90])
uses_memory=True
trials = 10
maps = [maps[0]]
brains = ["VisitedMemoryTabular"]
directions = {'00': 0, '10': 1, '-10': 2, '01': 3, '0-1': 4}

for mapp in maps:
    print(mapp[0])
    current_maze = mapp[1]
    maze_shape = np.shape(current_maze)
    maze_height = maze_shape[0]
    maze_width = maze_shape[1]
    for trial in range(0,trials+1):
        print(trial)
        #visited_counts = np.zeros((maze_height, maze_width), dtype=np.int32)
        visited_counts = np.zeros((maze_height, maze_width,num_actions+1),dtype=np.int32)
        reward_history = []
        trajectory_length_history = []
        qvalue_history = []
        goal_history = []
        Qtable2 = np.zeros((maze_height, maze_width, 2,4))
        epsilon_greedy = 0
        for iteration in range(iterations+1):
            time_step_history = 0
            total_reward = 0
            total_reward_eval = 0
            c = 0
            q_valuestep = []
            goal_states = mapp[3]
            goal_counter = 0
            #np.random.shuffle(goal_states)
            for goal_state in goal_states:
                visited_counts = visited_counts * 0
                # print("Starting trajectory with goal state",goal_state)
                state = mapp[2]
                done = False
                time_step = 0
                previous_action = 0
                action = 0
                direction_of_entry = 0
                next_direction_of_entry = direction_of_entry
                while not done:
                    # Choose an action
                    action = run_policy(state, epsilon_greedy,direction_of_entry)
                    # print("time_step",time_step,"state",state,"action",action)
                    next_state, reward, done , hitwall= environment_step(current_maze, action, state, goal_state)
                    # print("action",action_names[action], "next_state", next_state, "done", done, "Reward: ", reward)
                    difference = np.array(next_state) - np.array(state)
                    difference = ''.join([str(elem) for elem in difference])
                    next_direction_of_entry = directions[str(difference)]
                    apply_q_update(state, action, visited_counts, direction_of_entry, next_direction_of_entry, reward,
                                   next_state, done)
                    sy, sx = state
                    visited_counts[sy, sx, direction_of_entry] = 1  # +1 added by mike

                    state = next_state
                    total_reward += reward * (np.power(time_step, discount_factor))
                    previous_action = action
                    direction_of_entry = next_direction_of_entry
                    time_step += 1
                    if time_step >= (500):
                        done = True
            for goal_state in goal_states:
                # print("Starting trajectory with goal state",goal_state)
                state = mapp[2]
                done = False
                time_step = 0
                previous_action = 0
                action = 0
                visited_counts = visited_counts * 0
                direction_of_entry = 0
                next_direction_of_entry = direction_of_entry
                while not done:
                    # Choose an action
                    action = run_policy(state, 0.,direction_of_entry)
                    next_state, reward, done, hitwall = environment_step(current_maze, action, state, goal_state)
                    difference = np.array(next_state) - np.array(state)
                    difference = ''.join([str(elem) for elem in difference])
                    next_direction_of_entry = directions[str(difference)]
                    if next_state == goal_state:
                        goal_counter += 1
                    # print_transition_description(state,action,reward,state_)
                    sy, sx = state
                    visited_counts[sy, sx, direction_of_entry] = 1  # +1 added by mike
                    state = next_state
                    total_reward_eval += reward * (discount_factor ** time_step)
                    previous_action = action
                    direction_of_entry = next_direction_of_entry
                    time_step += 1
                    if time_step >= (500):
                        done = True
                time_step_history += time_step
            reward_history.append(total_reward_eval)
            trajectory_length_history.append(time_step_history)
            goal_history.append(goal_counter)
            if iteration % 500 == 0:
                print("Iteration "+str(iteration)+" Reward "+ str(trajectory_length_history[-1])+" Goal reached "+ str(goal_history[-1]))
                #print(Qtable2)
        #print(Qtable2)
        np.save("runs/"+str(trial) +"_map_"+mapp[0]+"_Brain_"+brains[0]+"Goals_reached.npy",np.array(goal_history))
        np.save("runs/"+str(trial) +"_map_"+mapp[0]+"_Brain_"+brains[0]+"_reward.npy",np.array(reward_history))
        np.save("runs/"+str(trial) + "_map_"+mapp[0]+"_Brain_"+ brains[0]+ "_step.npy", np.array(trajectory_length_history))

