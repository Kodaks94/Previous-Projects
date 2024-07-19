import random
import numpy as np
    
def environment_step(action,state,goal_state):
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
    if maze[new_y,new_x] == 1:
        # hit wall
        new_y=y
        new_x=x
    new_state = [new_y,new_x]
    reward = -1
    done = (new_state==goal_state)
    return new_state, reward, done

def run_policy(currentState, epsilon_greedy):
    if np.random.uniform() < epsilon_greedy:
        #choose random action
        choice = np.random.choice(range(len(action_names)))
    else: 
        #Greedy move
        q_values=Qtable[currentState[0],currentState[1],:,:]
        q_values=np.array([q_values[actions_memory[currentState[0],currentState[1],i],i] for i in range(num_actions)])
        best_q_value=q_values.max()
        best_q_indices=np.argwhere(q_values == best_q_value).flatten().tolist()
        #if same value choose random action
        choice = np.random.choice(best_q_indices)
    return choice

def apply_q_update(state, action, reward, next_state, done):
    sy,sx=state
    nsy,nsx=next_state
    current_q_value = Qtable[sy,sx,actions_memory[sy,sx,action],action]
    target_q_value = reward 
    if not done:
        future_state_q_values=Qtable[nsy,nsx,:, :]
        future_state_q_values=np.array([future_state_q_values[actions_memory[nsy,nsx,i],i] for i in range(num_actions)])
        target_q_value+= discount_factor * future_state_q_values.max()
    Qtable[sy,sx,actions_memory[sy,sx,action],action] += learning_rate * (target_q_value- current_q_value) #update
    return current_q_value, state
    
maze=np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,0,1,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1]])
		




maze_width=maze.shape[1]
maze_height=maze.shape[0]

action_names=["North","South","West","East"]
action_effects=[[-1,0],[1,0],[0,-1],[0,1]]
num_actions=len(action_names)
iterations = 2000
learning_rate = 0.1
epsilon_greedy = 0.1/5
discount_factor=1

start_state = [3,3] # center square
goal_states = [[1,3],[3,5],[5,3],[3,1]]
#ADDED NEW
start_state = complexlooped_start_state
goal_states = complexlooped_goal_states



# Create our table of Q-values.  We need 4 q-values for every cell in the maze.
num_memory_layers=2
Qtable = np.zeros((maze_height, maze_width, num_memory_layers, num_actions), dtype=np.float64)
actions_memory=np.zeros((maze_height, maze_width, num_actions), dtype=np.int32)

print("maze",maze)
print("start",start_state)
print("goals",goal_states)

uses_memory=True

reward_history=[]
trajectory_length_history=[]
for iteration in range(iterations):
    total_reward=0
    for goal_state in goal_states:
        state=start_state
        done=False
        time_step=0
        actions_memory=actions_memory*0
        while not done:
            # Choose an action
            action = run_policy(state, epsilon_greedy)
            #print("time_step",time_step,"state",state,"action",action)
            next_state, reward, done = environment_step(action, state, goal_state)
            #print("action",action, "next_state", next_state, "reward",reward, "done", done)
            
            apply_q_update(state, action, reward, next_state,  done)
            if uses_memory:
                sy,sx=state
                actions_memory[sy,sx,action]=1

            #print_transition_description(state,action,reward,state_)
            state = next_state
            total_reward += reward*(discount_factor**time_step)

            time_step+=1
            if time_step>500:
                done=True
    if iteration%10==0:
        print("Iteration",iteration,"Done.  Total_reward=",total_reward,"(Optimal = -32)")
    reward_history.append(total_reward)
    trajectory_length_history.append(time_step)


import matplotlib.pyplot as plt
plt.plot(reward_history)
plt.ylabel('Total Reward')
#plt.yscale('log')
plt.xlabel('Iteration')
plt.grid()
plt.show()
