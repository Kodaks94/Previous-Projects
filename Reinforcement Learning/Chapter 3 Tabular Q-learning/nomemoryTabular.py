import random
import sys

import numpy as np


def environment_step(tempmaze,action,state,goal_state):
    y = state[:, 0]
    x = state[:, 1]
    displacement=np.array([action_effects[a] for a in action])
    dx = displacement[:,0]
    dy = displacement[:,1]
    hit_wall = False
    new_x = x+dx
    new_y = y+dy
    tempmaze = tempmaze[0]
    new_state = np.array([[np.where(ny < 0 or ny >= maze_height or tempmaze[ny, nx] == 1, by, ny)
              ,np.where(nx <0 or nx>=maze_width or tempmaze[ny,nx] == 1,bx, nx)] for nx,ny,bx,by in zip(new_x,new_y,x,y)])
    reward = [-1] * trials
    done = np.max(np.where(new_state == goal_state, 1, 0),axis=1)
    return new_state, reward, done

def run_policy(currentState, ep):
    sy = currentState[:,0]
    sx = currentState[:,1]
    if np.random.uniform() < ep:
        choice = np.random.choice(range(len(action_names)),size=(trials,))
    else:
        q_values = [Qtable2[i,sy[i],sx[i],:] for i in range(trials)]
        best_q_value=[ q_value.max() for q_value in q_values]
        best_q_indices=[np.argwhere(q_value == best_q).flatten().tolist() for q_value, best_q in zip(q_values, best_q_value)]
        choice = [np.random.choice(best_q_indice) for best_q_indice in best_q_indices]
    return choice

def apply_q_update(state, action, reward, next_state, done):
    sy = state[:, 0]
    sx = state[:, 1]
    nsy = next_state[:,0]
    nsx = next_state[:,1]
    current_q_value = np.array([Qtable2[i, sy[i], sx[i], action[i]] for i in range(trials)])
    target_q_value = reward
    target_q_value = np.where(done == False, np.array([target_q_value[i] +discount_factor * Qtable2[i,nsy, nsx, :].max() for i in range(trials)]), reward)
    Qtable2[:,sy, sx,action]  = (Qtable2[:,sy, sx,action] + learning_rate * (target_q_value - current_q_value))
    return Qtable2

action_names=["North","South","West","East"]
action_effects=[[-1,0],[1,0],[0,-1],[0,1]]
num_actions=len(action_names)
iterations = 500 *4 * 10
learning_rate = 0.1
epsilon_greedy = 0
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
#complex_looped = np.genfromtxt("foo.csv",delimiter=',')
complexlooped_start_state = [10,10]
complexlooped_goal_states= [[1,1],[1,18],[18,1],[18,18]]
maps.append([name,complex_looped,complexlooped_start_state,complexlooped_goal_states,90])
uses_memory=False
trials = 10
brains = ["noMemoryTabular"]

for mapp in maps:
    print(mapp[0])
    current_maze = mapp[1]
    maze_shape = np.shape(current_maze)
    maze_height = maze_shape[0]
    maze_width = maze_shape[1]
    current_maze = np.array([current_maze] * trials)
    goal_states = mapp[3]
    Qtable2 = np.zeros((trials,maze_height, maze_width, num_actions), dtype=np.float64)
    epsilon_greedy = 0.1
    reward_history = []
    trajectory_length_history = []
    qvalue_history = []
    goal_history = []
    for iteration in range(iterations + 1):
        time_step_history = np.zeros((trials,1),dtype= float)
        total_reward =  np.zeros((trials,1),dtype=float)
        total_reward_eval = np.zeros((trials,1),dtype=float)
        goal_counter = np.zeros((trials,1),dtype=int)
        np.random.shuffle(goal_states)
        for goal_state in goal_states:
            state = np.array([mapp[2]]*trials)
            done = np.zeros([trials,], dtype=int)
            time_step = np.zeros([trials,],dtype=int)
            while np.sum(done) < trials:
                action = run_policy(state,epsilon_greedy)
                next_state, reward, done = environment_step(current_maze, action, state, goal_state)
                Qtable2 = apply_q_update(state, action, reward, next_state, done)
                state = next_state
                total_reward = total_reward+ reward
                previous_action = action
                time_step = time_step + np.where(time_step >= 500 + done, 0,1)
                outoftime = np.where(time_step >= 500,1,0)
                done = outoftime + done
        for goal_state in goal_states:
            state = np.array([mapp[2]] * trials)
            done = np.zeros([trials, ], dtype=int)
            time_step = np.zeros([trials, ], dtype=int)
            while np.sum(done) < trials:
                action = run_policy(state, epsilon_greedy)
                next_state, reward, done = environment_step(current_maze, action, state, goal_state)
                goal_counter = goal_counter + np.max(np.where(next_state == goal_state, 1, 0), axis=1)
                Qtable2 = apply_q_update(state, action, reward, next_state, done)
                state = next_state
                total_reward = total_reward + reward
                previous_action = action
                time_step = time_step + np.where(time_step >= 500 + done, 0, 1)
                outoftime = np.where(time_step >= 500, 1, 0)
                done = outoftime + done
            time_step_history = time_step_history + time_step
        reward_history.append(total_reward_eval)
        trajectory_length_history.append(time_step_history)
        goal_history.append(goal_counter)
    for i in range(trials):
        print("Trial "+ str(i)+" Reward "+ str(trajectory_length_history[i])+" Goal reached "+ str(goal_history[i]))
        np.save("runs/"+str(i) +"_map_"+mapp[0]+"_Brain_"+brains[0]+"Goals_reached.npy",np.array(goal_history[i]))
        np.save("runs/"+str(i) +"_map_"+mapp[0]+"_Brain_"+brains[0]+"_reward.npy",np.array(reward_history[i]))
        np.save("runs/"+str(i) + "_map_"+mapp[0]+"_Brain_"+ brains[0]+ "_step.npy", np.array(trajectory_length_history[i]))
