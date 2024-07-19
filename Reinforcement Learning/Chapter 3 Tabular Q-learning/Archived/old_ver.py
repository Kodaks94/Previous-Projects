import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import collections
import random
import sys


def build_q_network(n_features, time_steps, num_hidden_nodes, n_actions):
    inputLayer = tf.keras.layers.Input((time_steps, n_features))
    RNNLayer = tf.keras.layers.LSTM(num_hidden_nodes, activation='tanh')
    outputLayer = tf.keras.layers.Dense(n_actions, activation=None)
    return tf.keras.Sequential(name="LSTM", layers=
    [inputLayer, RNNLayer, outputLayer])


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
            s, a, r, s_, done = map(np.asarray, zip(replay_buffer[-1 - i]))
        samples_states = s if i == 0 else np.concatenate([s, samples_states], axis=0)
        samples_next_states = s_ if i == 0 else np.concatenate([s_, samples_next_states], axis=0)
        samples_actions = a if i == 0 else np.concatenate([a, samples_actions], axis=0)
        samples_r_values = r if i == 0 else np.concatenate([r, samples_r_values], axis=0)
        samples_done = done if i == 0 else np.concatenate([done, samples_done], axis=0)
    not_done_array = 1 - np.stack(samples_done)
    return samples_states, samples_next_states, samples_actions, samples_r_values.astype(
        np.float32), not_done_array.astype(np.float32)


@tf.function
def update_q_network(samples_states, samples_next_states, samples_actions, samples_r_values, not_done_array):
    # compute gradient:
    # print("Doing q update,sars_",s,a,r,s_)
    with tf.GradientTape() as tape:
        q_values = q_network(samples_states)
        # need to strip out the appropraiate (according to samples_actions) entry of each row here!
        q_values = tf.gather(q_values, indices=samples_actions, axis=1, batch_dims=1)  # this returns a tensor of rank 1
        q_targets = samples_r_values
        q_targets += discount_factor * tf.reduce_max(target_q_network(samples_next_states), axis=1) * not_done_array
        assert q_targets.shape == q_values.shape
        loss = mse_loss(tf.stop_gradient(q_targets), q_values)
    grads = tape.gradient(loss, q_network.trainable_variables)
    # print("Old q-values", q_network.get_weights())
    # print("Q-values used", q_values)
    # print("Q-values targets", q_targets)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    # print("New q-values", q_network.get_weights())
    # sys.exit(0)


def run_epsilon_greedy_policy(state, epsilon_greedy):
    assert len(state.shape) == 2
    assert state.shape[0] == time_sequence_length
    assert state.shape[1] == maze_height * maze_width  # input should be one-hot encoded!
    state = np.expand_dims(state, axis=0)  # add in dummy batch dimension
    # q_values = q_network(state)[0]
    q_values = run_q_network(state)  # not sure if this is any faster than the line above?
    assert len(q_values) == num_actions
    # Added increase epsilon for reducing randomness ->
    # self.epsilon *= eps_decay
    # self.epsilon = max(self.epsilon, eps_min)
    if np.random.random() < epsilon_greedy:
        return np.random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)


@tf.function
def run_q_network(state):
    return q_network(state)[0]


def one_hot_encode(state):
    encoded = np.zeros((maze_width * maze_height))
    encoded[state[0] * maze_height + state[1]] = 1
    return encoded


maps = []

name = 'smallCorridor'
SmallCorridor = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]])
SmallCorridor_goal_state = [[1, 3]]
SmallCorridor_start_state = [5, 1]
maps.append([name, SmallCorridor, SmallCorridor_start_state, SmallCorridor_goal_state, 3])

name = "Tshaped"
Tshaped = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]])
Tshaped_goal_states = [[5, 1], [5, 5]]
# Tshaped_goal_states = [[5,1]]
Tshaped_start_state = [1, 3]
maps.append([name, Tshaped, Tshaped_start_state, Tshaped_goal_states, 10])


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

name = 'complex'
complex = np.genfromtxt("complex(10,10).csv",delimiter=',')
complex_start_state = [1,1]
complex_goal_states= [[17,1],[19,19],[19,13],[17,13],[13,11]]
#maps.append([name,complex,complex_start_state,complex_goal_states,90])

algo_name = "RNNalgo"

chosen_maze = maps[2]
print("Using ", chosen_maze[0])
maze = chosen_maze[1]
start_state = chosen_maze[2]
goal_states = chosen_maze[3]
print("start_state", start_state)
print("goal_states", goal_states)
maze_width = maze.shape[1]
maze_height = maze.shape[0]
max_trajectory_length = 20
time_sequence_length = 4
reward_magnitued_per_timestep = 0.1

action_names = ["North", "South", "West", "East"]
action_effects = [[-1, 0], [1, 0], [0, -1], [0, 1]]
num_actions = len(action_names)

iterations = 1000
# learning_rate = 0.1
epsilon_greedy = 0.1
discount_factor = 1
num_hidden_nodes = 32
optimizer = tf.keras.optimizers.SGD(0.1)
# optimizer = tf.keras.optimizers.Adam()

use_separate_target_network = True
frequency_update_target_network = 1
frequency_update_q_network = 1
replay_chunk_size_to_use = 5
shuffle_replay_buffer = False
replay_buffer_size = 1000
replay_buffer = collections.deque(maxlen=replay_buffer_size)

q_network = build_q_network(maze_height * maze_width, time_sequence_length, num_hidden_nodes, num_actions)
if use_separate_target_network:
    target_q_network = build_q_network(maze_height * maze_width, time_sequence_length, num_hidden_nodes, num_actions)
else:
    target_q_network = q_network

mse_loss = tf.keras.losses.MeanSquaredError()

train_count = 0
for iteration in range(iterations):
    total_reward = 0
    total_steps = 0
    done = False
    for goal_state in goal_states:
        state = start_state
        state_history = np.array([one_hot_encode(state)] * time_sequence_length)

        done = False
        time_step = 0
        while not done:
            # Choose an action
            current_state_most_recent_chain = state_history[-time_sequence_length:,
                                              :] + 0  # note +0 forces a copy of this array so it can't be changed by something else!
            action = run_epsilon_greedy_policy(current_state_most_recent_chain, epsilon_greedy)
            next_state, reward, done = environment_step(action, state, goal_state)
            total_reward += reward * (discount_factor ** time_step)
            time_step += 1
            if time_step >= max_trajectory_length:
                done = True
            state_history = np.roll(state_history, -1, axis=0)
            state_history[-1] = one_hot_encode(next_state)
            next_state_most_recent_chain = state_history[-time_sequence_length:, :] + 0
            replay_buffer.append([current_state_most_recent_chain, action, reward * reward_magnitued_per_timestep,
                                  next_state_most_recent_chain, done])
            state = next_state
            current_state_most_recent_chain = next_state_most_recent_chain
            if len(replay_buffer) >= replay_chunk_size_to_use and time_step % frequency_update_q_network == 0:
                samples_states, samples_next_states, samples_actions, samples_r_values, not_done_array = fetch_episodes_from_replay_memory(
                    replay_buffer, replay_chunk_size_to_use, shuffle=shuffle_replay_buffer)
                update_q_network(samples_states, samples_next_states, samples_actions, samples_r_values, not_done_array)
                train_count += 1
                if train_count % frequency_update_target_network == 0 and target_q_network != q_network:
                    target_q_network.set_weights(q_network.get_weights())
        # print("training update completed")
    # print(state_history)
    print("Iteration ", iteration, "Completed ", time_step, "total reward", total_reward)
