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

def run_policy(currentState, ep,previous_action):
    sy,sx= currentState
    if np.random.uniform(0,1) < ep:
        choice = np.random.choice(range(len(action_names)))
    else:
        q_values=Qtable2[current_maze[sy,sx+1],current_maze[sy,sx-1], current_maze[sy+1,sx], current_maze[sy-1,sx],previous_action, :]
        assert len(q_values.shape)==1
        assert q_values.shape[0]==num_actions
        best_q_value=q_values.max()
        best_q_indices=np.argwhere(q_values == best_q_value).flatten().tolist()
        choice = np.random.choice(best_q_indices)
        assert choice>=0 and choice<num_actions
    return choice


def apply_q_update(state, action, previous_action, reward, next_state, done, timestep):
    sy,sx=state
    nsy,nsx=next_state
    current_q_value =Qtable2[current_maze[sy,sx+1],current_maze[sy,sx-1], current_maze[sy+1,sx], current_maze[sy-1,sx],previous_action, action]
    target_q_value =  reward
    if not done:
        future_state_q_values = Qtable2[current_maze[nsy,nsx+1],current_maze[nsy,nsx-1], current_maze[nsy+1,nsx], current_maze[nsy-1,nsx],action, :]
        assert len(future_state_q_values.shape)==1
        assert future_state_q_values.shape[0]==num_actions
        target_q_value += discount_factor * future_state_q_values.max()
    Qtable2[current_maze[sy,sx+1],current_maze[sy,sx-1], current_maze[sy+1,sx], current_maze[sy-1,sx],previous_action, action] += learning_rate * (target_q_value - current_q_value)  # update
    current_q_value = Qtable2[current_maze[sy, sx + 1], current_maze[sy, sx - 1], current_maze[sy + 1, sx], current_maze[sy - 1, sx], previous_action, action]
    return current_q_value, state

action_names=["North","South","West","East"]
action_effects=[[-1,0],[1,0],[0,-1],[0,1]]
num_actions=len(action_names)
iterations = 500 *4 * 10
#iterations = 5000
learning_rate = 0.1
discount_factor=.9
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
uses_memory=False
trials = 10
mapstrain = [maps[-2]]
brains = ["wallfollower"]


for mapp in mapstrain:
    print(mapp[0])
    current_maze = mapp[1]
    maze_shape = np.shape(current_maze)
    maze_height = maze_shape[0]
    maze_width = maze_shape[1]
    for trial in range(0,trials+1):
        print(trial)
        reward_history = []
        trajectory_length_history = []
        qvalue_history = []
        goal_history = []
        Qtable2 = np.zeros((2,2,2,2,4,4))
        epsilon_greedy = 0.1
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
                # print("Starting trajectory with goal state",goal_state)
                state = mapp[2]
                done = False
                time_step = 0
                previous_action = 0
                action = 0
                while not done:
                    # Choose an action
                    action = run_policy(state, epsilon_greedy, previous_action)
                    next_state, reward, done , hitwall= environment_step(current_maze, action, state, goal_state)
                    apply_q_update(state, action,previous_action, reward , next_state, done, time_step)
                    if next_state == goal_state:
                        goal_counter+=1

                    previous_action = action
                    state = next_state
                    total_reward += reward * (np.power(time_step, discount_factor))
                    time_step += 1
                    if time_step >= (500):
                        done = True
                time_step_history += time_step
        for valid_map in maps:
            print(valid_map[0])
            time_step_history = 0
            current_maze = valid_map[1]
            maze_shape = np.shape(current_maze)
            maze_height = maze_shape[0]
            maze_width = maze_shape[1]
            total_reward = 0
            total_reward_eval = 0
            c = 0
            q_valuestep = []
            goal_states = valid_map[3]
            goal_counter = 0
            for goal_state in goal_states:
                state = valid_map[2]
                done = False
                time_step = 0
                previous_action = 0
                action = 0
                while not done:
                    # Choose an action
                    action = run_policy(state, 0., previous_action)
                    # print("time_step",time_step,"state",state,"action",action)
                    next_state, reward, done, hitwall = environment_step(current_maze, action, state,
                                                                         goal_state)
                    previous_action = action
                    if next_state == goal_state:
                        goal_counter += 1
                    state = next_state
                    total_reward_eval += reward * (discount_factor ** time_step)
                    time_step += 1
                    if time_step >= (500):
                        done = True
                time_step_history += time_step
            reward_history.append(total_reward_eval)
            trajectory_length_history.append(time_step_history)
            goal_history.append(goal_counter)
            print("Iteration " + str(iteration) + " Reward " + str(
            trajectory_length_history[-1]) + " Goal reached " + str(goal_history[-1]))
            np.save("runs/"+str(trial) +"_map_"+valid_map[0]+"_Brain_"+brains[0]+"Goals_reached.npy",np.array(goal_counter))
            np.save("runs/"+str(trial) +"_map_"+valid_map[0]+"_Brain_"+brains[0]+"_reward.npy",np.array(total_reward_eval))
            np.save("runs/"+str(trial) + "_map_"+valid_map[0]+"_Brain_"+ brains[0]+ "_step.npy", np.array(time_step_history))