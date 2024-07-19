import numpy as np
import pandas as pd

# newQ(s,a) = oldQ(s,a) + (learning_rate * (reward + discounted factor * max(Q(s', a)) - oldQ(s,a)))
class TabularQLearning(object):

    def __init__(self, actions, learning_rate = 0.01, reward_decay = 1, e_greedy = 0.9,is_offpolicy = False):
        self.ep = e_greedy
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.actions = actions
        self.is_offpolicy = is_offpolicy

        self.Tab = pd.DataFrame(columns= self.actions, dtype= np.float64)



    def choose(self, currentState, is_ep):
        currentState = str(currentState)
        self.check_state_exist(currentState)
        if is_ep:

            if np.random.uniform() < self.ep :

            #Greedy move

                choice = self.Tab.loc[currentState, :]

            #if same value choose random action

                choice = np.random.choice(choice[choice == np.max(choice)].index)

            else:
                #choose random action
                choice = np.random.choice(self.actions)
        else:
            choice = self.Tab.loc[currentState, :]
            choice = np.random.choice(choice[choice == np.max(choice)].index)

        return choice

    def learn(self, s,a,r,s_,a_, done):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        p = self.Tab.loc[s,a]
        if not done:
            #terminal != s_ then next state
            if (self.is_offpolicy):
                target = r + self.reward_decay * self.Tab.loc[s_, :].max()
            else:
                target = r + self.reward_decay * self.Tab.loc[s_, a_]
        else:
            target = r
        self.Tab.loc[s,a] += self.learning_rate * (target- p) #update
        return self.Tab.loc[s], s
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
