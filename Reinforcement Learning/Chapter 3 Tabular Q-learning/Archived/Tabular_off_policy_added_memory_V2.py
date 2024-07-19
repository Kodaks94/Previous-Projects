import numpy as np
import pandas as pd
# newQ(s,a) = oldQ(s,a) + (learning_rate * (reward + discounted factor * max(Q(s', a)) - oldQ(s,a)))
# [x,y,memory,actions]
class TabularQLearning(object):

    def __init__(self, dimensions ,actions, learning_rate = 0.01, reward_decay = 1, e_greedy = 0.9, is_offpolicy = False):
        #dimensions [x,y,memory]
        self.dimensions = dimensions
        self.ep = e_greedy
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.actions = actions
        self.Table = np.zeros((dimensions[0],dimensions[1],dimensions[2],len(actions)))
        self.Memory = np.zeros((dimensions[0],dimensions[1]), dtype= int)
        self.is_offpolicy = is_offpolicy
        self.Tab = pd.DataFrame(columns= self.actions, dtype= np.float64)


    def reset_memory(self):
        self.Memory = np.zeros((self.dimensions[0],self.dimensions[1]))
    def choose(self,currentState, is_ep):
        #flip ep for standard
        if is_ep:
            if np.random.uniform() < self.ep:
                choice = self.Table[currentState[0],currentState[1],currentState[2],:]
            #choice = np.array([currentState[0],currentState[1],currentState[2],:])


                choice = np.random.choice(np.array(np.where(choice==np.max(choice)))[0])

            # Greedy: gather q-values, choose highest q-value, if there are more than one highest choose random between high qs
            else:
                choice = np.random.choice(len(self.actions))
        else:
            choice = self.Table[currentState[0],currentState[1],currentState[2],:]
            choice = np.random.choice(np.array(np.where(choice==np.max(choice)))[0])

        return choice
    def learn(self, s,a,r,s_, a_,done):

        self.check_state_exist(str(s))
        self.check_state_exist(str(s_))

        q_predicted = self.Table[s[0],s[1],s[2],a]
        if not done:
            if self.is_offpolicy:
                target = r + self.reward_decay* self.Table[s_[0],s_[1],s_[2],:].max()
            else:
                target = r + self.reward_decay * self.Table[s_[0], s_[1], s_[2], a_]

        else:
            target = r
        self.Table[s[0],s[1],s[2],a] += self.learning_rate*(target - q_predicted)
        #update
        q_value = [self.Table[s[0], s[1],1, :] , self.Table[s[0],s[1],0,:]]
        self.Memory[s[0], s[1]] = 1

        self.Tab.loc[str(s)]= q_value[s[2]]
        return q_value, s

    def check_state_exist(self, state):

        if state not in self.Tab.index:
            # append new state to q table
            self.Tab = self.Tab.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.Tab.columns,
                    name=state,
                )
            )