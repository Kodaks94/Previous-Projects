import random
import numpy as np
def environment_step(action, state, goal_state):
    y, x = state
    dy, dx = action_effects[action]
    new_x = x + dx
    new_y = y + dy
    if new_x < 0 or new_x >= maze_width:
        # off grid
        new_x = x
    if new_y < 0 or new_y >= maze_height:
        # off grid
        new_y = y
    if maze[new_y, new_x] == 1:
        # hit wall
        new_y = y
        new_x = x
    new_state = [new_y, new_x]
    reward = -1
    done = (new_state == goal_state)
    return new_state, reward, done

def run_policy(currentState, epsilon_greedy,previous_action):
    if np.random.uniform() < epsilon_greedy:
        # choose random action
        choice = np.random.choice(range(len(action_names)))
    else:
        # Greedy move
        q_values = Qtable[currentState[0], currentState[1], :, :]
        # q_values = Qtable[currentState[0], currentState[1], :, :,:]
        #q_values = np.array([q_values[actions_memory[currentState[0], currentState[1], 0 if action_memory_uses_state_only else i], previous_action,i] for i in range(num_actions)])
        q_values = np.array([q_values[previous_action,i] for i in range(num_actions)])
        best_q_value = q_values.max()
        best_q_indices = np.argwhere(q_values == best_q_value).flatten().tolist()
        # if same value choose random action
        choice = np.random.choice(best_q_indices)
    return choice
prev_action = 0
def apply_q_update(state, action, previous_action, reward, next_state, done):
    sy, sx = state
    nsy, nsx = next_state
    #current_q_value = Qtable[sy, sx, actions_memory[sy, sx, 0 if action_memory_uses_state_only else action], previous_action ,action]
    # current_q_value = Qtable[sy,sx,actions_memory[sy,sx,0 if action_memory_uses_state_only else action],action,previous_action]
    current_q_value = Qtable[sy, sx, previous_action,action]
    target_q_value = reward
    if not done:
        future_state_q_values = Qtable[nsy, nsx, :, :]
        # future_state_q_values = Qtable[nsy, nsx, :, :,:]
        #future_state_q_values = np.array([future_state_q_values[actions_memory[nsy, nsx, 0 if action_memory_uses_state_only else i], previous_action ,i] for i in range(num_actions)])
        # future_state_q_values = np.array([future_state_q_values[actions_memory[nsy, nsx, 0 if action_memory_uses_state_only else i], i,i] for i in range(num_actions)])
        future_state_q_values = np.array([future_state_q_values[previous_action, i] for i in range(num_actions)])
        # future_state_q_values=np.array([future_state_q_values[actions_memory[sy,sx,i],i] for i in range(num_actions)])
        target_q_value += discount_factor * future_state_q_values.max()
    #Qtable[sy, sx, actions_memory[sy, sx, 0 if action_memory_uses_state_only else action], previous_action,action] += learning_rate * ( target_q_value - current_q_value)  # update
    # Qtable[sy, sx, actions_memory[sy, sx, 0 if action_memory_uses_state_only else action], action,previous_action] += learning_rate * (target_q_value - current_q_value)  # update
    Qtable[sy, sx ,previous_action,action] += learning_rate * (target_q_value - current_q_value)  # update
    #print("Updated Q", state, action, actions_memory[sy, sx, 0 if action_memory_uses_state_only else action], "from",current_q_value, "to", Qtable[sy, sx, actions_memory[sy, sx, 0 if action_memory_uses_state_only else action]],"target", target_q_value)
    return current_q_value, state

maze_len = 3
maze = np.array([
    [0] * maze_len])
action_memory_uses_state_only = False
maze_width = maze.shape[1]
maze_height = maze.shape[0]
action_names = ["North", "South", "West", "East"]
action_effects = [[-1, 0], [1, 0], [0, -1], [0, 1]]
num_actions = len(action_names)
iterations = 500 * 4 * 4 * 10
learning_rate = 0.1
epsilon_greedy = 0.1
discount_factor = 0.1
start_state = [0, maze_len // 2]  # center square
goal_states = [[0, 0], [0, maze_len - 1]]
# Create our table of Q-values.  We need 4 q-values for every cell in the maze.
num_memory_layers = 2
Qtable = np.zeros((maze_height, maze_width,num_actions+1 ,num_actions),
                dtype=np.float64)  # this will include x,y,num_layer, num_action, num_prev_action
# Qtable = np.zeros((maze_height, maze_width, num_memory_layers, num_actions,num_actions), dtype=np.float64)
# Qtable = np.zeros((maze_height, maze_width,num_actions,num_actions), dtype=np.float64)
actions_memory = np.zeros((maze_height, maze_width, 1 if action_memory_uses_state_only else num_actions),
                        dtype=np.int32)
import sys
print(np.shape(Qtable))
print("maze", maze)
print("start", start_state)
print("goals", goal_states)
uses_memory = True
reward_history = []
trajectory_length_history = []
# Qtable-=10
for iteration in range(iterations + 1):
    total_reward = 0
    timestep_history = 0
    if iteration >= iterations - 10:
        epsilon_greedy = 0
    for goal_state in goal_states:
        # print("Starting trajectory with goal state",goal_state)
        state = start_state
        done = False
        time_step = 0
        actions_memory = actions_memory * 0
        previous_action = 0
        while not done:
            # Choose an action
            action = run_policy(state, epsilon_greedy,previous_action)
            # print("time_step",time_step,"state",state,"action",action)
            next_state, reward, done = environment_step(action, state, goal_state)
            # print("action",action_names[action], "next_state", next_state, "done", done)
            apply_q_update(state, action, previous_action, reward, next_state, done)
            if uses_memory:
                sy, sx = state
                actions_memory[sy, sx, 0 if action_memory_uses_state_only else action] = 1
            # print_transition_description(state,action,reward,state_)
            state = next_state
            total_reward += reward * (discount_factor ** time_step)
            previous_action = action+1
            time_step += 1
            if time_step > 40:
                done = True
        timestep_history += time_step
    print("Iteration", iteration, "Done.  Total_reward=", timestep_history, "(Optimal =",
          (maze_len // 2) * 2 + (maze_len - 1))
    # print(Qtable)
    reward_history.append(total_reward)
    trajectory_length_history.append(time_step)

