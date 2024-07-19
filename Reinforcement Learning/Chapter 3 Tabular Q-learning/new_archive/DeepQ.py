import random
import sys
import tensorflow as tf
import numpy as np
import collections

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

def build_q_network(n_features, time_steps, num_hidden_nodes, n_actions):
    inputLayer = tf.keras.layers.Input((time_steps,n_features))
    #ForwardLayer = tf.keras.layers.Dense(num_hidden_nodes,activation = 'relu')
    RNNLayer = tf.keras.layers.LSTM(num_hidden_nodes, activation='tanh')
    outputLayer= tf.keras.layers.Dense(n_actions, activation=None)
    return tf.keras.Sequential(name="LSTM",layers=
             [inputLayer,RNNLayer,outputLayer])
def one_hot_encode(state):
    encoded = np.zeros((maze_width*maze_height))
    encoded[state[0]* maze_height + state[1]] = 1
    return encoded

@tf.function
def run_q_network(state):
    return  q_network(state)[0]
def run_epsilon_greedy_policy(state, epsilon_greedy):
    assert len(state.shape) == 2
    assert state.shape[0] == time_sequence_length
    assert state.shape[1] == maze_height * maze_width  # input should be one-hot encoded!
    state = np.expand_dims(state, axis=0)  # add in dummy batch dimension
    # q_values = q_network(state)[0]
    q_values = run_q_network(state)  # not sure if this is any faster than the line above?
    q_values = np.reshape(q_values, (4,))

    assert len(q_values) == num_actions
    # Added increase epsilon for reducing randomness ->
    # self.epsilon *= eps_decay
    # self.epsilon = max(self.epsilon, eps_min)
    if np.random.random() < epsilon_greedy:
        return np.random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)
def fetch_episodes_from_replay_memory(replay_buffer, number_of_replay_samples_to_use, shuffle=True):
    # TODO can function this be sped up by avoiding the loop, and just using slice notation to get the answer in 1 step?
    samples_states = []
    samples_r_values = []
    samples_next_states = []
    samples_done = []
    samples_actions = []
    for i in range(number_of_replay_samples_to_use):
        if shuffle:
            s, a, r, s_, done = map(np.asarray, zip(*random.sample(replay_buffer, 1)))
        else:
            s, a, r, s_, done = map(np.asarray, zip(replay_buffer[-1-i]))
        samples_states = s if i ==0 else np.concatenate([s,samples_states], axis=0)
        samples_next_states = s_ if i ==0 else np.concatenate([s_,samples_next_states], axis=0)
        samples_actions = a if i ==0 else np.concatenate([a, samples_actions], axis =0)
        samples_r_values = r if i ==0 else np.concatenate([r, samples_r_values], axis =0)
        samples_done = done if i ==0 else np.concatenate([done, samples_done], axis =0)
    not_done_array=1-np.stack(samples_done)
    return samples_states,samples_next_states,samples_actions,samples_r_values.astype(np.float32),not_done_array.astype(np.float32)
@tf.function
def update_q_network(samples_states,samples_next_states,samples_actions,samples_r_values,not_done_array):
    # compute gradient:
    #print("Doing q update,sars_",s,a,r,s_)
    with tf.GradientTape() as tape:
        q_values = q_network(samples_states)
        # need to strip out the appropraiate (according to samples_actions) entry of each row here!
        q_values=tf.gather(q_values, indices=samples_actions,axis=1, batch_dims=1) # this returns a tensor of rank 1
        q_targets = samples_r_values
        q_targets+= discount_factor* tf.reduce_max(target_q_network(samples_next_states),axis=1)*not_done_array
        assert q_targets.shape == q_values.shape
        loss = mse_loss(tf.stop_gradient(q_targets), q_values)
    grads = tape.gradient(loss,q_network.trainable_variables)
    #print("Old q-values", q_network.get_weights())
    #print("Q-values used", q_values)
    #print("Q-values targets", q_targets)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    #print("New q-values", q_network.get_weights())
    #sys.exit(0)


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
uses_memory=True
trials = 10
optimizer = tf.keras.optimizers.SGD(learning_rate)

reward_magnitued_per_timestep = 0.1
time_sequence_length = 500
use_separate_target_network = True
brains = ["DeepQLearning"]
#maps = [maps[-2]]
for mapp in maps:
    print(mapp[0])
    current_maze = mapp[1]
    maze_shape = np.shape(current_maze)
    maze_height = maze_shape[0]
    maze_width = maze_shape[1]
    goal_states = mapp[3]
    time_sequence_length = 1 if not uses_memory else 8 * (len(goal_states) - 1)
    for trial in range(0,trials+1):
        print(trial)
        reward_history = []
        trajectory_length_history = []
        qvalue_history = []
        goal_history = []
        q_network = build_q_network(maze_height * maze_width, time_sequence_length, 12, num_actions)
        if use_separate_target_network:
            target_q_network = build_q_network(maze_height * maze_width, time_sequence_length, 12,
                                               num_actions)
        mse_loss = tf.keras.losses.MeanSquaredError()
        total_recorded_history = []
        total_reward_recorded_history = []
        reward_over_iteration = []
        use_separate_target_network = True
        frequency_update_target_network = 1
        frequency_update_q_network = 1
        replay_chunk_size_to_use = 5
        shuffle_replay_buffer = False
        replay_buffer_size = 50
        replay_buffer = collections.deque(maxlen=replay_buffer_size)
        epsilon_greedy = 0.1
        train_count = 0
        for iteration in range(iterations+1):
            if iteration >= iterations / 2:
                epsilon_greedy = 0
            time_step_history = 0
            total_reward = 0
            total_reward_eval = 0
            c = 0
            q_valuestep = []
            goal_states = mapp[3]
            goal_counter = 0
            np.random.shuffle(goal_states)
            for goal_state in goal_states:
                # print("Starting trajectory with goal state",goal_state)
                state = mapp[2]
                state_history = np.array([one_hot_encode(state)] * time_sequence_length)
                done = False
                time_step = 0
                recorded_history = []
                while not done:
                    # Choose an action
                    recorded_history.append(state)
                    current_state_most_recent_chain = state_history[-time_sequence_length:,
                                                      :] + 0  # note +0 forces a copy of this array so it can't be changed by something else!
                    action = run_epsilon_greedy_policy(current_state_most_recent_chain,
                                                       epsilon_greedy)

                    # print("time_step",time_step,"state",state,"action",action)
                    next_state, reward, done , hitwall= environment_step(current_maze, action, state, goal_state)
                    # print("action",action_names[action], "next_state", next_state, "done", done, "Reward: ", reward)

                    state_history = np.roll(state_history, -1, axis=0)
                    state_history[-1] = one_hot_encode(next_state)
                    next_state_most_recent_chain = state_history[-time_sequence_length:, :] + 0
                    replay_buffer.append(
                        [current_state_most_recent_chain, action, reward * reward_magnitued_per_timestep,
                         next_state_most_recent_chain, done])
                    current_state_most_recent_chain = next_state_most_recent_chain
                    if len(replay_buffer) >= replay_chunk_size_to_use and time_step % frequency_update_q_network == 0:
                        samples_states, samples_next_states, samples_actions, samples_r_values, not_done_array = fetch_episodes_from_replay_memory(
                            replay_buffer, replay_chunk_size_to_use, shuffle=shuffle_replay_buffer)
                        update_q_network(samples_states, samples_next_states, samples_actions, samples_r_values,
                                         not_done_array)
                        train_count += 1
                        if train_count % frequency_update_target_network == 0 and target_q_network != q_network:
                            target_q_network.set_weights(q_network.get_weights())

                    # print_transition_description(state,action,reward,state_)
                    state = next_state
                    total_reward += reward * (discount_factor ** time_step)
                    previous_action = action
                    time_step += 1
                    if time_step >= (500):
                        done = True
                #time_step_history += time_step
            for goal_state in goal_states:
                # print("Starting trajectory with goal state",goal_state)
                state = mapp[2]
                state_history = np.array([one_hot_encode(state)] * time_sequence_length)
                done = False
                time_step = 0
                recorded_history = []
                while not done:
                    # Choose an action
                    recorded_history.append(state)
                    current_state_most_recent_chain = state_history[-time_sequence_length:,
                                                      :] + 0  # note +0 forces a copy of this array so it can't be changed by something else!
                    action = run_epsilon_greedy_policy(current_state_most_recent_chain,
                                                       epsilon_greedy)
                    # print("time_step",time_step,"state",state,"action",action)
                    next_state, reward, done, hitwall = environment_step(current_maze, action, state, goal_state)
                    if next_state == goal_state:
                        goal_counter += 1
                    # print("action",action_names[action], "next_state", next_state, "done", done, "Reward: ", reward)
                    # print_transition_description(state,action,reward,state_)
                    state = next_state
                    total_reward += reward * (discount_factor ** time_step)
                    previous_action = action
                    time_step += 1
                    if time_step >= (500):
                        done = True
                time_step_history += time_step
            reward_history.append(total_reward_eval)
            trajectory_length_history.append(time_step_history)
            goal_history.append(goal_counter)
            print("Iteration "+str(iteration)+" Reward "+ str(trajectory_length_history[-1])+" Goal reached "+ str(goal_history[-1]))
        np.save("runs/"+str(trial) +"_map_"+mapp[0]+"_Brain_"+brains[0]+"Goals_reached.npy",np.array(goal_history))
        np.save("runs/"+str(trial) +"_map_"+mapp[0]+"_Brain_"+brains[0]+"_reward.npy",np.array(reward_history))
        np.save("runs/"+str(trial) + "_map_"+mapp[0]+"_Brain_"+ brains[0]+ "_step.npy", np.array(trajectory_length_history))

