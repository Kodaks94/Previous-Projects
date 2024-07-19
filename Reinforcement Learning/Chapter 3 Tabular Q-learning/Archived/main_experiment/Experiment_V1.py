from  Brains.Tabular_off_policy_added_memory_V3 import TabularQLearning as rl2
from Brains.RL_off_policy_Tabular_brain import TabularQLearning as rl1
from  Environments.dynamic_maze_v1 import dynamic_maze as maze
from Brains.dqn_brain_simple import DQN as deeprl
from Brains.RL_off_policy_dqn_brain import model as DQN2
from  Utility.Vector2D import Vector2D
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import numpy as np
import os
import csv
from itertools import zip_longest
import tensorflow.keras as keras





def update():

    use_ep = True
    average_rewards = []
    iter = []
    print("Starting brain ", brain)
    for iteration in range(iterations):
        iter.append(iteration)
        t_w_ = 0
        t_w_e = 0
        step_ = 0
        step_e = 0
        f_h_, f_g_, r_s_ = [False,False,False]
        f_h_e, f_g_e, r_s_e = [False, False, False]
        action = 0

        if iteration % 500 == 0:
           print(iteration)


        for episode in range(episodes):
            # initial observation
            state, goal_pos = env.reset()
            env.render()
            if brain == 3 or brain ==4:
                RL.reset_memory()
            while True:
                env.render()
                current_state = Vector2D.convert_to_numbers_raw(state, env.units)

                if brain == 3 or brain == 4:
                    #print(current_state[0], current_state[1],action,RL.Memory[current_state[0], current_state[1],action])
                    current_state.append(int(RL.Memory[current_state[0], current_state[1],action]))
                action = RL.choose(current_state, True )

                # RL take action and get next observation and reward
                state_, reward, done, coords, steps, distance, f_h, f_g, r_s, goal_pos = env.step(action)

                step_ = steps
                next_state = Vector2D.convert_to_numbers_raw(state_, env.units)
                if brain == 3 or brain ==4:
                    next_state.append(int(RL.Memory[next_state[0], next_state[1],action]))

                # RL learn from this transition
                next_action = RL.choose(next_state, True)
                q_value, s = RL.learn(current_state, action, reward, next_state,next_action, done)
                if brain == 3 or brain == 4:
                    s = [s[0], s[1]]
                #if current_state == [3, 1, 0] or current_state == [3,1,1]:
                #    print(current_state, q_value)

                #if brain == 2:
                env.update_qtable(s, q_value[0], q_value[1])
                #else:
                    #env.update_qtable(s,q_value,None)
                state = state_
                prev_action = action
                t_w_e += reward
                if (done):
                    f_h_e = f_h
                    f_g_e = f_g
                    r_s_e = r_s
                    break

        for episode in range(episodes):
            # initial observation
            state, goal_pos = env.reset()
            env.render()
            if brain == 3 or brain ==4:
                RL.reset_memory()
            while True:
                env.render()
                current_state = Vector2D.convert_to_numbers_raw(state, env.units)

                if brain == 3 or brain == 4:
                    current_state.append(int(RL.Memory[current_state[0], current_state[1],action]))
                action = RL.choose(current_state, False )

                # RL take action and get next observation and reward
                state_, reward, done, coords, steps, distance, f_h, f_g, r_s, goal_pos = env.step(action)
                step_e = steps
                next_state = Vector2D.convert_to_numbers_raw(state_, env.units)
                if brain == 3 or brain ==4:
                    next_state.append(int(RL.Memory[next_state[0], next_state[1],action]))

                # RL learn from this transition
                next_action = RL.choose(next_state, False)
                #q_value, s = RL.learn(current_state, action, reward, next_state,next_action, done)
                #if brain == 3 or brain == 4:
                #    s = [s[0], s[1]]
                #if current_state == [3, 1, 0] or current_state == [3,1,1]:
                #    print(current_state, q_value)

                #if brain == 2:
                #env.update_qtable(s, q_value[0], q_value[1])
                #else:
                    #env.update_qtable(s,q_value,None)
                state = state_
                prev_action = action
                t_w_ += reward
                if (done):
                    f_h_ = f_h
                    f_g_ = f_g
                    r_s_ = r_s

                    break

        data.append([iteration, current_map, current_AI, is_memory, TabularQ, t_w_/episodes, step_, f_g_, f_h_, r_s_ , t_w_e/ episodes,step_e, f_g_e, f_h_e, r_s_e, goal_pos, RL.reward_decay,max_steps])
        #if iteration % 100 == 0:
            #print("iteraton", iteration,"brain", brain, "current_map", current_map, t_w)
        #print("Average reward for iteration:",iteration, " is ",t_w/episodes)

    #plt.ion()
    #plt.plot(iter, average_rewards, '-b', label='rewards over episodes')
    #plt.xlabel('episode')
    #plt.ylabel('rewards')
    #plt.ioff()
    #plt.show()
    #comments = input("Any Comments?")
    #np.save(logpath+comments+'experiment_V1_Tshaped_memory', Experiment_log)
    #plt.savefig(logpath + classname + comments + "Graph" + ".png")
    #input("Enter to end")
    env._snapsaveCanvas(logpath + "Tablememv1" + ".png")
    #input("set it to be over")
    #print('game over')

    RL.Tab.to_csv(current_map+"_"+current_AI+".csv")
    env.destroy()

def dqn_v1_update():
    total_rewards = []
    iter = []
    step = 0
    print("starting brain ",brain)
    for iteration in range(iterations):
        iter.append(iteration)
        t_w = 0

        for episode in range(episodes):

            state, _ = env.reset()
            state = state [:2]
            distance = env.give_distance(state)
            #state.append(env.goalpos)

            state = np.reshape(state, (1, RL.n_features))
            if brain ==6 or brain==8:
                state = np.reshape(np.append(state, RL.rnn_memory_size*[0]), (1,RL.n_features+RL.rnn_memory_size))



            while True:
                env.render()

                action, action_value, memory_value = RL.choose_action(state)


                state_, reward, done, coords, steps, distance,f_h, f_g, r_s, goal_pos = env.step(action)
                state_ , _ = state_
                state_ = state_[:2]
                #state_.append(env.goalpos)


                state_ = np.reshape(state_, (1,RL.n_features))
                if brain == 6 or brain == 8:
                    state_ = np.reshape(np.append(state_, memory_value), (1,RL.n_features + RL.rnn_memory_size))

                step += 1

                t_w += reward[1]

                #if action_value is not None:
                    #env.update_qtable(env.convert_to_numbers(coords), action_value[0])


                if brain == 5 or brain == 6:
                    RL.store_transition(state, action, reward[0], state_)
                    if (step > 200) and (step % 5 == 0):
                        RL.learn(state, action, reward[0], state_, False)
                else:
                    RL.learn(state, action, reward[0], state_, True)

                state = state_
                if done:
                    break
        #if iteration % 100 == 0:
            #print("iteraton", iteration, "brain", brain, "current_map", current_map, t_w)
        data.append([iteration, current_map, current_AI, is_memory, TabularQ, t_w/episodes,steps, f_g,f_h, r_s, t_w/episodes,steps, f_g,f_h, r_s,goal_pos, RL.gamma, max_steps])
    #env.reset()
    #RL.sess.close()

    env.destroy()
    return


if __name__ == "__main__":


    '''Iteration,map_name,algorithm_name,Algorithm_has_memory,TabularQ, reward'''

    maps_name = [

        'smallCorridor2goal',
        'LongCorridorMaze',
        'crossmap',
        'Tshaped',

    ]
    maps_name = [


        'crossmap'


    ]



    steps  = [
        50,
        50,
        50,
        50,
    ]

    algorithm_types =  [
        'off policy',
        'DQN',
    ]



    data = []
    is_memory = False
    TabularQ = False
    current_AI = ''
    current_map = ''
    max_steps = 11
    iterations = 20000
    episodes = 1
    path = os.path.dirname(__file__).split("/")
    print(path)
    mazepath = path[:]
    logpath = path[:]
    logpath.append('Logs')
    mazepath.append('maps')
    mazepath = ''.join([str(elem) + "/" for elem in mazepath])
    logpath = ''.join([str(elem) + "/" for elem in logpath])
    #brains:: offtabnomemory, ontabnomemory, offtabmemory, ontabnomemory, deepnomemory, deepmemory
    # deepnomemorywithrepmem, deepmemorywithmem
    brains = [3]

    for i in range(1):

        print("TRIAL ", i)
        for brain in brains:
            for map_index in range(len(maps_name)):
                current_map = maps_name[map_index]
                max_steps = steps[map_index]
                if brain == 1:
                    env = maze(mazepath + maps_name[map_index], max_steps, False)
                    RL = rl1(actions= list(range(env.n_actions)), is_offpolicy= True)
                    current_AI = "OFFPolicyTabularQNoMemory"
                    is_memory = False
                    TabularQ = True

                    env.after(100, update)
                    env.mainloop()

                elif brain ==2:
                    env = maze(mazepath + maps_name[map_index], max_steps, False)
                    RL = rl1(actions=list(range(env.n_actions)), is_offpolicy=False)
                    current_AI = "ONPolicyTabularQNoMemory"
                    is_memory = False
                    TabularQ = True
                    env.after(100, update)
                    env.mainloop()

                elif brain == 3:
                    env = maze(mazepath + maps_name[map_index], max_steps, False)
                    current_AI = "OFFPolicyTabularQMemory"
                    memory_size = 2
                    if current_map == '4goal':
                        memory_size = 4
                    RL = rl2([env.maze_w, env.maze_h, memory_size], actions=list(range(env.n_actions)),is_offpolicy=True)
                    is_memory = True
                    TabularQ = True
                    env.after(100, update)
                    env.mainloop()

                elif brain == 4:
                    env = maze(mazepath + maps_name[map_index], max_steps, False)
                    current_AI = "ONPolicyTabularQMemory"
                    memory_size = 2
                    if current_map == '4goal':
                        memory_size = 4
                    RL = rl2([env.maze_w, env.maze_h, memory_size], actions=list(range(env.n_actions)), is_offpolicy=False)
                    is_memory = True
                    TabularQ = True
                    env.after(100, update)
                    env.mainloop()
                elif brain==5:
                    env = maze(mazepath + maps_name[map_index], max_steps, True)
                    current_AI = "OFFPolicyDeepQNoMemory"
                    RL = DQN2(env.n_actions, 2,
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              memory_size=2000,
                              output_graph=False,
                              rnn_memory_size=0
                              )
                    is_memory = True
                    TabularQ = False
                    env.after(100, dqn_v1_update)
                    env.mainloop()
                    RL.sess.close()
                    del RL
                elif brain == 6:
                    env = maze(mazepath + maps_name[map_index], max_steps, True)
                    current_AI = "OFFPolicyDeepQWithMemory"
                    RL = DQN2(env.n_actions, 2,
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              memory_size=2000,
                              output_graph=False,
                              rnn_memory_size= 2
                              )
                    is_memory = True
                    TabularQ = False
                    env.after(100, dqn_v1_update)
                    env.mainloop()
                    RL.sess.close()
                    del RL
                elif brain==7:
                    env = maze(mazepath + maps_name[map_index], max_steps, True)
                    current_AI = "OFFPolicyDeepQNoMemoryWithReplayMemory"
                    RL = DQN2(env.n_actions, 2,
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              memory_size=2000,
                              output_graph=False,
                              rnn_memory_size=0
                              )
                    is_memory = True
                    TabularQ = False
                    env.after(100, dqn_v1_update)
                    env.mainloop()
                    RL.sess.close()
                    del RL
                elif brain == 8:
                    env = maze(mazepath + maps_name[map_index], max_steps, True)
                    current_AI = "OFFPolicyDeepQWithMemorywithReplayMemory"
                    RL = DQN2(env.n_actions, 2,
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              memory_size=2000,
                              output_graph=False,
                              rnn_memory_size= 5
                              )
                    is_memory = True
                    TabularQ = False
                    env.after(100, dqn_v1_update)
                    env.mainloop()
                    RL.sess.close()
                    del RL

        #rows=  zip(data[0], data[1], data[2],data[3],data[4],data[5],data[6],data[7],data[8])
        print(np.shape(data))
        with open("experiment_results_trials_e_change_"+str(i)+".csv", "w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['iteration','map_name','algorithm_name','Algorithm_has_memory','TabularQ','reward_no_e', 'steps_no_e', 'finished_in_goal_no_e', 'finished_in_hell_no_e', 'ran_out_of_steps_no_e','reward_with_e', 'steps_with_e', 'finished_in_goal_with_e', 'finished_in_hell_with_e', 'ran_out_of_steps_with_e', 'goal_position', 'reward_decay', 'max_steps'])
            for row in data:
                writer.writerow(row)

        data = []
        f.close()
