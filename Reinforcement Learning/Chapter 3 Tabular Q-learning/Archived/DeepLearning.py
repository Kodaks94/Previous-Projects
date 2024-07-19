import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque


class DQN(Model):

    def __init__(self, units,n_actions):
        super(DQN, self).__init__()
        self.layers = []
        for i in units:
            self.layers.append(tf.keras.layers.Dense(i,activation='relu'))
        self.layers.append(tf.keras.layers.Dense(n_actions))

    def call(self, state):
        out = state
        for layer in self.layers:
            layer(out)
        return out

#TODO ADD RDQN CLASS
class DQRN(Model):

    def __init__(self, units, n_actions):
        super(DQRN, self).__init__()
        for i in units:
            self.layers.append(tf.keras.layers.LSTM(i,activation='relu'))
        self.layers.append(tf.keras.layers.Dense(n_actions))
    def call(self, state):
        out = state
        for layer in self.layers:
            layer(out)
        return out


class agent:
    def __init__(self):
        self.USE_REPLAY_MEMORY = True
        self.lr = 0.001
        self.gamma = 0.99
        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = optimizers.Adam(lr=self.lr)
        self.batch_size = 4
        self.state_size = 2
        self.action_size= 4
        #REPLAYMEMORY
        if self.USE_REPLAY_MEMORY:
            self.memory = deque(maxlen=2000)

    def target_update(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        q_value = self.dqn_model(tf.convert_to_tensor([state], dtype= tf.float32))
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action, q_value

    #REPLAY MEMORY FUNCTIONS
    def store_transition(self, s,a,r,s_,done):
        self.memory.append((s,a,r,s_,done))

    def update(self,state,action,reward,next_state,dones):
        if self.USE_REPLAY_MEMORY:
            batch = np.random.sample(self.memory,self.batch_size)
            s,a,r,s_,dones = [[i[j] for i in batch] for j in range(len(batch[0]))]
        else:
            s,r,s_,dones = [state,action,reward,next_state,dones]
        dqn_variables = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variables)
            r = tf.convert_to_tensor(r, dtype=tf.float32)
            a = tf.convert_to_tensor(a, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(s_),dtype=tf.float32))
            a_ = tf.argmax(target_q, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(a_,self.action_size)*target_q, axis=1)
            target_value = (1-dones) * self.gamma * target_value + r
            current_q = self.dqn_model(tf.convert_to_tensor(np.vstack(s),dtype=tf.float32))
            current_value = tf.reduce_sum(tf.one_hot(a,self.action_size)*current_q, axis=1)
            sqr_difference = tf.reduce_mean(tf.square(current_value- target_value)*0.5)
        dqn_gradients = tape.gradient(sqr_difference,dqn_variables)
        self.opt.apply_gradients(zip(dqn_gradients,dqn_variables))









