import random
import numpy as np
#import tensorflow.compat.v1 as tf
# Tabular with/without memory
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
#

#DQN with/without memory
def build(n_features,n_actions, rnn_memory_size, gamma ,lr):
    # ------------------ all inputs ------------------------
    n_actions += rnn_memory_size
    s = tf.placeholder(tf.float32, [None, n_features + rnn_memory_size], name='s')  # input State
    s_ = tf.placeholder(tf.float32, [None, n_features + rnn_memory_size], name='s_')  # input Next State
    r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
    a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
    # ------------------ build evaluate_net ------------------
    e1 = tf.layers.dense(s, 25, tf.nn.relu, kernel_initializer=w_initializer,
                         bias_initializer=b_initializer)
    q_eval = tf.layers.dense(e1, n_actions, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer)
    # ------------------ build target_net ------------------
    t1 = tf.layers.dense(s_, 25, tf.nn.relu, kernel_initializer=w_initializer,
                         bias_initializer=b_initializer)
    q_next = tf.layers.dense(t1, n_actions, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer)
    q_target = r + gamma * tf.reduce_max(q_next, axis=1, name='Qmax_s_')  # shape=(None, )
    # needed
    # qval moving towards to qval, qval estimate not true, stop gradient pretend this is an estimate, dont do gradient descent
    q_target = tf.stop_gradient(q_target)
    a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
    q_eval_wrt_a = tf.gather_nd(params=q_eval, indices=a_indices)  # shape=(None, )
    loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_wrt_a, name='TD_error'))
    _train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    # add Gradient to see progress
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    with tf.variable_scope('hard_replacement'):
        target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
    # self.sess = tf.Session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return sess, s,a,r,s_,n_features, rnn_memory_size

def learn(s,a,r,s_, sess, _train_op, loss, _s, _a, _r, _s_, n_features, rnn_memory_size):
    obs = np.hstack((s, [[a, r]], s_))
    _, cost = sess.run(
        [_train_op, loss], feed_dict={
            _s: obs[:, :n_features + rnn_memory_size],
            _a: obs[:, n_features + rnn_memory_size],
            _r: obs[:, n_features + rnn_memory_size + 1],
            _s_: obs[:, -1 * (n_features + rnn_memory_size):],
        }
    )
    #epsilon = epsilon + epsilon_increment if epsilon < epsilon_max else epsilon_max

def choose_action(observation, sess, s, q_eval,action_memory_pointer, epsilon, n_actions):
    actions_value = sess.run(q_eval, feed_dict={s: np.array(observation)})

    memory_value = actions_value[:, action_memory_pointer:]

    if np.random.uniform() < epsilon:
        # forward feed the observation and get q value for every actions

        action = np.argmax(actions_value[:, :action_memory_pointer])
    else:
        action = np.random.randint(0, n_actions)

    actions_value = np.concatenate([[action], np.asarray(memory_value[0])])
    return action, actions_value, memory_value
#
maps = []

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

#
name = 'complex'
complex = np.genfromtxt("complex(10,10).csv",delimiter=',')
complex_start_state = [1,1]
complex_goal_states= [[17,1],[19,19],[19,13],[17,13],[13,11]]
#maps.append([name,complex,complex_start_state,complex_goal_states,90])

#
name = 'complex_looped'
complex_looped = np.genfromtxt("foo.csv",delimiter=',')
complexlooped_start_state = [10,10]
complexlooped_goal_states= [[1,1],[1,18],[18,1],[18,18]]
maps.append([name,complex_looped,complexlooped_start_state,complexlooped_goal_states,90])
algo_name = "memoryTab"


uses_memory = True
history_total_rewards = []
for map in maps:
    # CHOOSE MAZE
    maze = map[1]
    start_state = map[2]
    goal_states = map[3]
    #

    maze_width = maze.shape[1]
    maze_height = maze.shape[0]

    action_names = ["North", "South", "West", "East"]
    action_effects = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    num_actions = len(action_names)
    iterations = 2000
    learning_rate = 0.1
    epsilon_greedy = 0.1 / 5
    discount_factor = 1
    # Create our table of Q-values.  We need 4 q-values for every cell in the maze.
    num_memory_layers = 2
    Qtable = np.zeros((maze_height, maze_width, num_memory_layers, num_actions), dtype=np.float64)
    actions_memory = np.zeros((maze_height, maze_width, num_actions), dtype=np.int32)
    print("maze", map[0])
    #print("maze", maze)
    print("start", start_state)
    print("goals", goal_states)

    state_history = []
    reward_history = []
    trajectory_length_history = []
    for iteration in range(iterations):
        if iteration == iterations-1:
             epsilon_greedy = 0
        total_reward = 0
        for goal_state in goal_states:
            state = start_state
            done = False
            time_step = 0
            actions_memory = actions_memory * 0
            if iteration == iterations - 1:
                goal_state_history = []
            while not done:
                if iteration == iterations - 1:
                    goal_state_history.append(state)
                # Choose an action
                action = run_policy(state, epsilon_greedy)
                # print("time_step",time_step,"state",state,"action",action)
                next_state, reward, done = environment_step(action, state, goal_state)
                # print("action",action, "next_state", next_state, "reward",reward, "done", done)

                apply_q_update(state, action, reward, next_state, done)
                if uses_memory:
                    sy, sx = state
                    actions_memory[sy, sx, action] = 1

                # print_transition_description(state,action,reward,state_)
                state = next_state
                total_reward += reward * (discount_factor ** time_step)
                time_step += 1
                if time_step > maze_width*maze_height:
                    done = True
            if iteration == iterations - 1:
                state_history.append(goal_state_history)
                history_total_rewards.append([algo_name,map[0],total_reward])
        if iteration % 10 == 0:
            print("Iteration", iteration, "Done.  Total_reward=", total_reward, "(Optimal = "+str(map[-1])+")")

        reward_history.append(total_reward)
        trajectory_length_history.append(time_step)



    count = 0
    for history in state_history:
        goal = goal_states[count]
        #maze[goal[0], goal[1]] = 5
        count += 1
        for s in history:
            maze[s[0]][s[1]] = 8

    np.savetxt("maze_" + map[0]+"_"+ algo_name+ ".csv", maze, delimiter=',', fmt='%1.0f')


    import matplotlib.pyplot as plt
    plt.plot(reward_history)
    plt.legend([i[0] for i in maps])
    plt.ylabel('Total Reward')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.grid()

name = "maze_experiment"+"_"+algo_name
plt.savefig(fname=name)



import csv
file = open('overal_results.csv', 'w+', newline ='')
with file:
    header = ['algoname', 'mapname','total_reward']
    reader = csv.reader(file)
    write = csv.writer(file)
    if next(reader,None) != header:
        write.writerow(header)
    for data in history_total_rewards:
        write.writerow(data)











