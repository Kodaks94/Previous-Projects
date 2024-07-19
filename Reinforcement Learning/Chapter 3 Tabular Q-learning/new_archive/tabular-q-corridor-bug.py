import random
import sys

import numpy as np


def environment_step(tempmaze,action,state,goal_state):
    y,x = state
    dy,dx=action_effects[action]
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
    new_state = [new_y,new_x]
    reward = -1
    done = (new_state==goal_state)
    return new_state, reward, done

def run_policy(currentState, epsilon_greedy,previous_action):
    sy,sx= currentState
    if np.random.uniform() < epsilon_greedy:
        #choose random action
        choice = np.random.choice(range(len(action_names)))
    else:
        #Greedy move
        if Test == 1:
            q_values=Qtable1[sy,sx,:,:]
            q_values=np.array([q_values[visited_counts[sy,sx,0 if action_memory_uses_state_only else i],i] for i in range(num_actions)])
            #q_values=np.array([q_values[visited_counts[sy,sx,i],i] for i in range(num_actions)])
        elif Test == 2:
             q_values=Qtable2[sy,sx,previous_actions[sy,sx,0],:]
        elif Test == 3:
            q_values = Qtable3[sy, sx, :, :,:]
            q_values = np.array([q_values[visited_counts[sy, sx, 0 if action_memory_uses_state_only else i], previous_action,i]
              for i in range(num_actions)])
        assert len(q_values.shape)==1
        assert q_values.shape[0]==num_actions
        best_q_value=q_values.max()
        best_q_indices=np.argwhere(q_values == best_q_value).flatten().tolist()
        #if same value choose random action
        choice = np.random.choice(best_q_indices)
        assert choice>=0 and choice<num_actions
    return choice

def apply_q_update(state, action,previous_action, reward, next_state, done):
    sy,sx=state
    nsy,nsx=next_state
    if Test ==1:
        current_q_value = Qtable1[sy, sx, visited_counts[sy, sx, 0 if action_memory_uses_state_only else action], action]
        #current_q_value = Qtable[sy, sx, visited_counts [sy, sx,action], action]
    elif Test ==2:
        current_q_value = Qtable2[sy, sx, previous_actions[sy, sx, 0], action]
    elif Test == 3:
        current_q_value = Qtable3[sy,sx,visited_counts[sy,sx,0 if action_memory_uses_state_only else action],previous_action,action]
    target_q_value = reward
    if not done:
        if Test ==1:
            future_state_q_values=Qtable1[nsy,nsx,:, :]
            future_state_q_values=np.array([future_state_q_values[visited_counts[nsy,nsx,0 if action_memory_uses_state_only else i],i] for i in range(num_actions)])
            #future_state_q_values=np.array([future_state_q_values[visited_counts [nsy,nsx,i],i] for i in range(num_actions)])
        elif Test ==2:
            future_state_q_values=Qtable2[nsy,nsx,:,:]
            future_state_q_values=np.array([future_state_q_values[previous_actions[nsy,nsx,0],i] for i in range(num_actions)])
            #print("previous action of the cell: ",previous_actions[nsy, nsx, 0])
        elif Test == 3:
            future_state_q_values = Qtable3[nsy, nsx, :, :,:]
            future_state_q_values = np.array([future_state_q_values[visited_counts[nsy, nsx, 0 if action_memory_uses_state_only else i], action+1,i] 
            for i in range(num_actions)])
        assert len(future_state_q_values.shape)==1
        assert future_state_q_values.shape[0]==num_actions
       # print(">>>",future_state_q_values.max(), discount_factor, target_q_value,target_q_value + (discount_factor * future_state_q_values.max()))
        #print("target_q_value", target_q_value)
        target_q_value += discount_factor * future_state_q_values.max()
        #print("discount_factor",discount_factor)
        #print("future_state_q_values.max",future_state_q_values.max())
        #print("discount_factor * future_state_q_values.max()",discount_factor * future_state_q_values.max())
        #print("target_q_value after update",target_q_value)

    if Test == 1:
        Qtable1[sy, sx, visited_counts[sy, sx, 0 if action_memory_uses_state_only else action],action] += learning_rate * (target_q_value - current_q_value)  # update
    elif Test == 2:
        #print("Qtable_before:",Qtable2[sy,sx,previous_actions[sy,sx,0],action])
        Qtable2[sy,sx,previous_actions[sy,sx,0],action] += learning_rate * (target_q_value- current_q_value) #update
        #print("learning_rate * (target_q_value- current_q_value)",learning_rate * (target_q_value- current_q_value))
        #print("Qtable_After:",Qtable2[sy,sx,previous_actions[sy,sx,0],action])
        #print("Updated Q", sy,sx,previous_actions[sy,sx,0],action , action_names[action], "from", current_q_value, "to", Qtable2[sy,sx,previous_actions[sy,sx,0],action], "target",target_q_value)
        current_q_value ==  Qtable2[sy,sx,previous_actions[sy,sx,0],action]

    elif Test == 3:
        Qtable3[sy, sx, visited_counts[sy,sx,0 if action_memory_uses_state_only else action],previous_action, action] += learning_rate * (target_q_value - current_q_value)  # update
        #print("Updated Q", sy, sx, visited_counts[sy, sx, 0 if action_memory_uses_state_only else action],
        #      previous_action, action, state, action_names[action], "from", current_q_value, "to", Qtable3[
        #          sy, sx, visited_counts[
        #              sy, sx, 0 if action_memory_uses_state_only else action], previous_action, action], "target",
        #      target_q_value)
        current_q_value == Qtable3[
                  sy, sx, visited_counts[
                      sy, sx, 0 if action_memory_uses_state_only else action], previous_action, action]


    return current_q_value, state

action_memory_uses_state_only=True
action_names=["North","South","West","East"]
action_effects=[[-1,0],[1,0],[0,-1],[0,1]]
#action_names=["West","East"] # if we uncomment these 2 lines, then all 3 algorithms start working.
#action_effects=[[0,-1],[0,1]]
num_actions=len(action_names)
iterations = 500 * 4 * 4 * 10
learning_rate = 0.1
epsilon_greedy = 0.1
discount_factor= 0.99

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

num_memory_layers = 2
uses_memory=True
trials = 1
brains = ["SimpleTabular"]
maps = [maps[0]]


def run_exp(ep,vc,pa, rh, th):
    time_step_history = 0
    total_reward = 0
    c = 0
    q_valuestep = []
    goal_states = mapp[3]
    #np.random.shuffle(goal_states)
    for goal_state in goal_states:
        # print("Starting trajectory with goal state",goal_state)
        state = mapp[2]
        done = False
        time_step = 0
        previous_action = 0
        v_c = vc * 0
        p_a = pa * 0
        while not done:
            # Choose an action
            action = run_policy(state, ep, previous_action)
            # print("time_step",time_step,"state",state,"action",action)
            next_state, reward, done = environment_step(current_maze, action, state, goal_state)
            # print("action",action_names[action], "next_state", next_state, "done", done, "Reward: ", reward)
            apply_q_update(state, action, p_a, reward, next_state, done)
            if uses_memory:
                sy, sx = state
                v_c[sy, sx, 0 if action_memory_uses_state_only else action] = 1
                p_a[sy, sx, 0] = action + 1  # +1 added by mike
            # print_transition_description(state,action,reward,state_)
            state = next_state
            total_reward += reward * (discount_factor ** time_step)
            previous_action = action
            time_step += 1
            if time_step >= (20):
                done = True
        time_step_history += time_step
    if ep==0:
        rh.append(total_reward)
        th.append(time_step_history)
    return rh, th
for mapp in maps:
    print(mapp[0])
    current_maze = mapp[1]
    maze_shape = np.shape(current_maze)
    maze_height = maze_shape[0]
    maze_width = maze_shape[1]
    for brain in brains:
        print(brain)
        if brain == "SimpleTabular":
            Test = 1
            uses_memory = True
            num_memory_layers = 2
        elif brain == "MemoryTabular":
            Test = 2
            uses_memory = True
            print("passed")
        for trial in range(0,trials+1):
            print(trial)
            visited_counts = np.zeros((maze_height, maze_width, 1 if action_memory_uses_state_only else num_actions),
                                      dtype=np.int32)
            previous_actions = np.zeros((maze_height, maze_width, 1), dtype=np.int32)
            reward_history = []
            trajectory_length_history = []
            qvalue_history = []
            if Test == 1:
                Qtable1 = np.zeros((maze_height, maze_width, num_memory_layers, num_actions), dtype=np.float64)
            elif Test == 2:
                Qtable2 = np.zeros((maze_height, maze_width, num_actions + 1, num_actions), dtype=np.float64)
            elif Test == 3:
                Qtable3 = np.zeros((maze_height, maze_width, num_memory_layers, num_actions + 1, num_actions),
                                   dtype=np.float32)
            for iteration in range(iterations+1):
                reward_history, trajectory_length_history = run_exp(0.1,visited_counts,previous_actions,reward_history, trajectory_length_history)
                reward_history, trajectory_length_history = run_exp(0,visited_counts,previous_actions,reward_history, trajectory_length_history)
                print("Iteration "+str(iteration)+" Reward "+ str(trajectory_length_history[-1]))
            #np.save("runs/q_current_simple",np.array(qvalue_history,dtype = object))
            np.save("runs/"+str(trial) +"_map_"+mapp[0]+"_Brain_"+brain+"_reward.npy",np.array(reward_history))
            np.save("runs/"+str(trial) + "_map_"+mapp[0]+"_Brain_"+ brain+ "_step.npy", np.array(trajectory_length_history))

