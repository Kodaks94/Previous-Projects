import sys

import gym
from gym import spaces
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.logger import Logger
class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.passed_maze = maze
        self.current_maze = maze[1]
        self.maze_shape = np.shape(self.current_maze)
        self.maze_height = self.maze_shape[0]
        self.maze_width = self.maze_shape[1]
        self.current_state = np.concatenate([maze[2],[0,0,0,0]])
        #self.current_state = maze[2]
        self.max_timestep = 500
        self.goal_states = maze[3]
        self.time_step = 0
        self.reward_range = (-1, -1)
        self.goal_counter = 0
        self.action_names = ["North", "South", "West", "East"]
        self.action_effects = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.num_actions = len(self.action_names)
        #self.observation_space = spaces.MultiDiscrete(np.concatenate([[2]*self.maze_height*self.maze_width,[2,2,2,2]]))
        self.observation_space = spaces.MultiDiscrete([self.maze_height,self.maze_width,2,2,2,2])
        #self.observation_space = spaces.MultiDiscrete([self.maze_height , self.maze_width])
        print(self.observation_space)
        #[2] * self.maze_height * self.maze_width
        self.action_space = spaces.Discrete(self.num_actions)
        self.goal_state = self.goal_states[0]
        self.visited_counts = np.zeros((self.maze_height, self.maze_width,self.num_actions),dtype=np.int32)


    def step(self, action):
        y, x = self.current_state[:2]
        done = False
        dy, dx = self.action_effects[action]
        hit_wall = False
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x >= self.maze_width:
            # off grid
            new_x = x
        if new_y < 0 or new_y >= self.maze_height:
            # off grid
            new_y = y
        if int(self.current_maze[new_y, new_x]) == 1:
            # hit wall
            new_y = y
            new_x = x
            hit_wall = True
        new_state = [new_y, new_x]
        reward = -1
        done = (new_state == self.goal_state)
        self.time_step += 1
        if self.time_step >= self.max_timestep:
            done = True
        self.visited_counts[y, x, action] = 1
        new_state = np.concatenate([new_state,[self.visited_counts[new_y,new_x,0],
                                              self.visited_counts[new_y,new_x,1],
                                              self.visited_counts[new_y, new_x, 2],
                                              self.visited_counts[new_y, new_x, 3]
                                              ]])

        observ_state = (new_y-1)*self.maze_height + (new_x-1)
        a  = [observ_state]
        #a = [0] * self.maze_height * self.maze_width
        #a[observ_state] = 1
        #
        #observation = np.concatenate([a,[self.visited_counts[new_y,new_x,0],
        #                             self.visited_counts[new_y,new_x,1],
        #                                      self.visited_counts[new_y, new_x, 2],
        #                                      self.visited_counts[new_y, new_x, 3]
        #                                      ]])
        observation = a
        self.current_state = new_state
        return np.array(new_state), reward, done, {}

    def reset(self):
        #self.visited_counts = self.visited_counts *  0
        current_state = self.passed_maze[2]
        current_state = np.concatenate([[self.passed_maze[2][0]],[self.passed_maze[2][1]],[0,0,0,0]])
        observ_state = (current_state[0]-1)*self.maze_height + (current_state[1]-1)
        a = [observ_state]
        #a = [0] * self.maze_height * self.maze_width
        #a[observ_state] = 1
        #observation = np.concatenate([a, [0, 0, 0, 0]])
        observation = a
        self.current_state = current_state
        #np.random.shuffle(self.goal_states)
        self.goal_state = self.goal_states[self.goal_counter%len(self.goal_states)]
        self.time_step = 0
        self.goal_counter +=1
        return np.array(current_state)

maps = []
name = 'smallCorridor'
SmallCorridor = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]])

SmallCorridor_goal_state = [[3, 2], [3, 4]]
SmallCorridor_start_state = [3, 3]
maps.append([name, SmallCorridor, SmallCorridor_start_state, SmallCorridor_goal_state, 3])

#
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
LongCorridor = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]])
LongCorridor_goal_state = [[1, 3], [3, 3]]
LongCorridor_start_state = [1, 1]
maps.append([name, LongCorridor, LongCorridor_start_state, LongCorridor_goal_state, 18])
#
name = 'crossmap'
cross = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]])
cross_start_state = [3, 3]  # center square
cross_goal_states = [[1, 3], [3, 5], [5, 3], [3, 1]]
maps.append([name, cross, cross_start_state, cross_goal_states, 12])
'''
#
'''
name = 'complex_looped'
complex_looped = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])
height = len(complex_looped)
width = len(complex_looped[0])

# complex_looped = np.genfromtxt("foo.csv",delimiter=',')
complexlooped_start_state = [10, 10]
complexlooped_goal_states = [[1, 1], [1, 18], [18, 1], [18, 18]]
maps.append([name, complex_looped, complexlooped_start_state, complexlooped_goal_states, 90])
maps = [maps[1]]
size = maps[0][1]

print(len(size[0])*len(size))
iterations = 500 *4 * 10*5
# iterations = 5000
learning_rate = 0.001
epsilon_greedy = 0
discount_factor = .9
trials = 1
policy = "MlpPolicy"
brains = [A2C,PPO,DQN]
from stable_baselines3.common.env_checker import check_env
#logger = Logger("\runs","csv")
for trial in range(trials):
    for mapp in maps:
        for brain in brains:
            goal_states = mapp[3]
            map_name = mapp[0]
            env = MazeEnv(mapp)
            eval_env = MazeEnv(mapp)
            print("Trial_"+str(trial) + "_map_onehotencode_" + str(mapp[0]) + "_Brain_memory_" + str(brain.__name__))
            if brain.__name__ == "DQN":
                model = brain(policy, env, verbose=1,learning_rate=learning_rate,gamma=discount_factor,exploration_initial_eps=0.1,exploration_final_eps=0.,device='cuda')
            elif brain.__name__ == "PPO":
                model = brain(policy, env, verbose=1, learning_rate=learning_rate, gamma=discount_factor,device='cuda')
            elif brain.__name__ == "A2C":
                model = brain(policy, env, verbose=1, learning_rate=learning_rate, gamma=discount_factor,device='cuda')

            print(brain,model.policy)
            #model.set_logger(logger)
            #model.learn(total_timesteps=iterations, log_interval=iterations)
            #total_timestep = []
            #for goal in goal_states:
            #    eval_env.goal_state = goal
            #    done = False
            #    timestep = 0
            #    while not done:
            #        a = model.policy.predict(eval_env.current_state)
            #        new_state, reward, done, _ = eval_env.step(a[0])
            #        timestep +=1
            #    eval_env.reset()
            #    total_timestep.append(timestep)
            #print(total_timestep)
            #np.save("runs/Trial_" + str(trial) + "_map_" + str(mapp[0]) + "_Brain_Memory_" + str(brain.__name__) + "_step.npy",
            #        np.array(total_timestep))
#
