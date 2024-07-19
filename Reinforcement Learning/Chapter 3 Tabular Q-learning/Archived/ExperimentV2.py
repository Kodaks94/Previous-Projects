import numpy as np
import os
import tkinter as tk
import csv
import tkinter.font as tkf
import sys
#from PIL import ImageGrab #no linux comp

Graphics = True
Graphics_show_Q = True
maps_name = [

        'smallCorridor2goal',
        'LongCorridorMaze',
        'crossmap',
        'Tshaped',

    ]
maps_name = [
        'complex(10,10)'
    ]
steps  = [
        50,
        50,
        50,
        50,
    ]
# brains::
#
# [1: Tab No memory
# [2: Tab with Memory V1
# [3: Tab with Memory V2
brains = [3]


'''
    BRAINS
'''
class TabularQLearningWithoutMemory(object):
    def __init__(self, dimensions, actions, learning_rate=0.01, reward_decay=1, e_greedy=0.9, is_offpolicy=False):
        # dimensions [x,y,memory]
        self.dimensions = dimensions
        self.ep = e_greedy
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.actions = actions
        self.Table = np.zeros((dimensions[0], dimensions[1], len(actions)))
        self.is_offpolicy = is_offpolicy

    def choose(self,currentState, is_ep):
        #flip ep for standard
        if is_ep:
            if np.random.uniform() < self.ep:
                choice = self.Table[currentState[0],currentState[1],:]
            #choice = np.array([currentState[0],currentState[1],currentState[2],:])
                choice = np.random.choice(np.array(np.where(choice==np.max(choice)))[0])
            # Greedy: gather q-values, choose highest q-value, if there are more than one highest choose random between high qs
            else:
                choice = np.random.choice(len(self.actions))
        else:
            choice = self.Table[currentState[0],currentState[1],:]
            choice = np.random.choice(np.array(np.where(choice==np.max(choice)))[0])
        return choice
    def learn(self, s,a,r,s_, a_,done):
        q_predicted = self.Table[s[0],s[1],a]
        if not done:
            if self.is_offpolicy:
                target = r + self.reward_decay* self.Table[s_[0],s_[1],:].max()
            else:
                target = r + self.reward_decay * self.Table[s_[0], s_[1], a_]
        else:
            target = r
        self.Table[s[0],s[1],a] += self.learning_rate*(target - q_predicted)
        #update
        q_value = self.Table[s[0], s[1], :]
        return q_value, s

class TabularQLearningWithMemoryV1(object):

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
        return q_value, s

class TabularQLearningWithMemoryV2(object):

    def __init__(self, dimensions ,actions, learning_rate = 0.01, reward_decay = 1, e_greedy = 0.9, is_offpolicy = False):
        #dimensions [x,y,memory]
        self.dimensions = dimensions
        self.ep = e_greedy
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.actions = actions
        self.Table = np.zeros((dimensions[0],dimensions[1],dimensions[2],len(actions)))
        self.Memory = np.zeros((dimensions[0],dimensions[1],len(actions)), dtype= int)
        self.is_offpolicy = is_offpolicy
    def reset_memory(self):
        self.Memory = np.zeros((self.dimensions[0],self.dimensions[1],len(self.actions)), dtype= int)
    def choose(self,currentState, is_ep):
        #flip ep for standard
        if not(is_ep) or np.random.uniform() < self.ep:
            # Greedy: gather q-values, choose highest q-value, if there are more than one highest choose random between high qs
            choices_across_memory_options = self.Table[currentState[0],currentState[1],:,:]
            choice=np.array([choices_across_memory_options[self.Memory[currentState[0], currentState[1], a_index],a_index] for a_index in range(len(self.actions))])
            #print("choice",choice)
            choice = np.random.choice(np.array(np.where(choice==np.max(choice)))[0])
        else:
            choice = np.random.choice(len(self.actions))
        return choice

    def learn(self, s,a,r,s_, a_,done):
        memory_level_chosen=self.Memory[s[0], s[1], a]
        q_predicted = self.Table[s[0],s[1],memory_level_chosen,a]
        next_state_q_values_across_memory_options=self.Table[s_[0],s_[1],:,:]
        next_state_q_values=np.array([next_state_q_values_across_memory_options[self.Memory[s_[0], s_[1], a_index],a_index] for a_index in range(len(self.actions))])
        if not done:
            if self.is_offpolicy:
                target = r + self.reward_decay* next_state_q_values.max()
            else:
                target = r + self.reward_decay * self.Table[s_[0], s_[1], s_[2], a_]
                raise Exception("Don't want it to take this option ever")
        else:
            target = r
        self.Table[s[0],s[1],memory_level_chosen,a] += self.learning_rate*(target - q_predicted)
        #update
        q_value = [self.Table[s[0], s[1],1, :] , self.Table[s[0],s[1],0,:]]
        self.Memory[s[0], s[1], a] = 1
        return q_value, s


'''
    ENVIRONMENT CODE
'''
class Vector2D(object):


    def __init__(self, x,y):

        self.x = float(x)
        self.y = float(y)

    def get(self):
        return [self.x, self.y]
    def set(self, vector):
        self.x = vector.x
        self.y = vector.y
    def convert_to_coords(self, units):
        unit = units / 2
        row_coords = self.y * units + unit
        column_coords = self.x * units + unit
        return [column_coords, row_coords, column_coords + 30, row_coords + 30]
    def get_center_coords(self, units):
        number_ = self.convert_to_numbers(units)
        unit = units /2
        x_coords = number_[0] * units + unit
        y_coords = number_[1] * units + unit
        return [x_coords, y_coords]
    def convert_to_numbers(self, units):
        #unit = units / 2
        row_number = (self.y ) / units
        column_number = (self.x ) / units
        return [int(column_number), int(row_number)]

    def convert_to_numbers_raw(state,units):
        # unit = units / 2

        x,y, e,h = state
        row_number = (y) / units
        column_number = (x) / units
        return [int(column_number), int(row_number)]
class dynamic_maze(tk.Tk, object):

    def __init__(self, maze_name, maxsteps, is_deep):
        try:
            if maze_name[-4] != ".npy":
                maze_name += ".npy"
            data = np.load( maze_name, allow_pickle=True)
        except IOError as e:
            raise


        spec, self.hells, self.agents, self.goals, self.labelled_dict_list = data
        self.next_goal_index=0
        self.maze_w, self.maze_h, self.units = spec
        #print(spec)
        self.width = self.maze_w * self.units
        self.height = self.maze_h * self.units
        self.h_pixels = (self.maze_h - 1) * self.units
        self.w_pixels = (self.maze_w - 1) * self.units
        #print(self.h_pixels, self.w_pixels)
        super(dynamic_maze, self).__init__()
        self.title = maze_name
        self.geometry('{0}x{1}'.format(self.width, self.height))
        self.maxsteps = maxsteps
        self.is_deep = is_deep
        self.steps = 0
        self.oval = None
        self.build()
        self.Q_value_presentation()

        self.action_space = ['up', 'down', 'right', 'left']
        self.n_actions = len(self.action_space)
        #print(self.labelled_dict_list)


    def build_Astar_label(self):
        raise("Shouldn't be using this")
        labelled_maze = []
        for i in range(self.maze_h):
            row_vals = []
            for j in range(self.maze_w):
                label = lambda : 0 if ( not self.does_list_exist(self.hells,[j,i])) else 1
                row_vals.append(label())
            labelled_maze.append(row_vals)
        return labelled_maze

    def Q_value_presentation(self):
        #unit = (self.units - 30) / 2
        self.q_table = {}

        helv36 = tkf.Font(family='Helvetica',
                             size=5)
        for y in range(self.maze_h):
            for x in range(self.maze_w):
                vector = Vector2D(x,y)
                coords = vector.convert_to_coords(self.units)
                self.q_table[str(x) + str(y)] = self.canvas.create_text(coords[0], coords[1], text="0",font=helv36)

    def build(self):

        self.canvas = tk.Canvas(self, height=self.height, width=self.width, bg='white')
        for c in range(0, self.width, self.units):
            line = c, 0, c, self.height
            self.canvas.create_line(line[0], line[1], line[2], line[3])
        for r in range(0, self.height, self.units):
            line = 0, r, self.width, r
            self.canvas.create_line(line[0], line[1], line[2], line[3])
        for hell in self.hells:
            self.add_hell(Vector2D(hell[1][0], hell[1][1]))
        self.add_agent()
        self.add_goal()
        self.canvas.pack()

    def _snapsaveCanvas(self, filename):
        self.canvas.delete(self.rect)
        self.canvas.delete(self.oval)
        self.render()
        canvas = self._canvas()  # Get Window Coordinates of Canvas
        self.grabcanvas = ImageGrab.grab(bbox=canvas).save(filename)
    def _canvas(self):

        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        box = (x, y, x1, y1)

        return box
    def get_state(self):

        if self.is_deep:
            temp = (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                    self.maze_w * self.units)

            temp = self.canvas.coords(self.rect)

            return temp, self.goalpos
        else:
            current = self.canvas.coords(self.rect)
        return current, self.goalpos

    def reset(self):
        self.update()
        self.canvas.delete(self.rect)
        self.add_agent()
        self.add_goal()
        self.steps = 0
        return self.get_state()

    def step(self, action):
        s = self.canvas.coords(self.rect)
        b_a = np.array([0,0])
        if action == 0:  # up
            if s[1] > self.units:
                b_a[1] -= self.units
        elif action == 1:  # down
            if s[1] < self.h_pixels:
                b_a[1] += self.units
        elif action == 2:  # right
            if s[0] < self.w_pixels:
                b_a[0] += self.units
        elif action == 3:  # left
            if s[0] > self.units:
                b_a[0] -= self.units
        if action != 4:  # stop
            self.canvas.move(self.rect, b_a[0], b_a[1])
        self.steps += 1
        next_state = self.canvas.coords(self.rect)
        converted_state = self.get_state()

        self.finished_in_hell = False
        self.finished_in_goal = False
        self.Ran_out_of_steps = False
        c_s = Vector2D(next_state[0], next_state[1]).convert_to_numbers(self.units)
        if self.does_list_exist(self.hells, c_s):
            next_state = s
            converted_state = [s, self.goalpos]
            self.canvas.move(self.rect, -b_a[0], -b_a[1])

        if self.is_deep:
            reward, done, distance = self.DQN_reward_function2(Vector2D(next_state[0], next_state[1]))
            #converted_state = (np.array(next_state[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
            #            self.maze_w * self.units)
            reward = [-1,-1]
            return converted_state, reward, done, next_state, self.steps, distance, self.finished_in_hell, self.finished_in_goal,self.Ran_out_of_steps, self.goalpos
        else:
            reward, done, distance = self.experimentfunc2(Vector2D(next_state[0], next_state[1]))
            reward = -1
            return converted_state[0], reward, done, next_state, self.steps, distance, self.finished_in_hell, self.finished_in_goal, self.Ran_out_of_steps, self.goalpos


    def experimentfunc2(self, vector):
        state_labelled = vector.convert_to_numbers(self.units)
        if (state_labelled[0], state_labelled[1]) in self.labelled_dict_list[self.goalpos]:
            distance = self.labelled_dict_list[self.goalpos][state_labelled[0], state_labelled[1]]
        else:
            distance = self.maxsteps

        if state_labelled == self.goals[self.goalpos][2]:
            self.finished_in_goal = True
            reward = self.maxsteps
            done = True
        elif (self.steps >= self.maxsteps):
            self.Ran_out_of_steps = True
            reward = -1
            done = True
        elif self.does_list_exist(self.hells, state_labelled):
            self.finished_in_hell = True
            reward = -2
            done = True
        else:
            reward = -1
            done = False

        return reward, done, distance
    def give_distance(self, vector):
        vector = Vector2D(vector[0], vector[1])
        state_labelled = vector.convert_to_numbers(self.units)
        if (state_labelled[0], state_labelled[1]) in self.labelled_dict_list[self.goalpos]:
            distance = self.labelled_dict_list[self.goalpos][state_labelled[0], state_labelled[1]]
            return distance
        else:
            distance = self.maxsteps
            return distance


    def DQN_reward_function2(self, vector):
        raise Exception("Shouldn't be using this")
        state_labelled = vector.convert_to_numbers(self.units)
        if (state_labelled[0], state_labelled[1]) in self.labelled_dict_list[self.goalpos]:
            distance = self.labelled_dict_list[self.goalpos][state_labelled[0], state_labelled[1]]
        else:
            distance = self.maxsteps
        if state_labelled == self.goals[self.goalpos][2]:
            reward = [self.maxsteps,self.maxsteps]
            self.finished_in_goal = True
            done = True

        elif self.does_list_exist(self.hells, state_labelled):
            reward =[-2, - 2]
            self.finished_in_hell = True
            done = True
        elif self.steps >= self.maxsteps:
            self.Ran_out_of_steps = True
            reward = [-distance, -distance]
            done = True
        else:
            reward = [-1,-1]
            done = False
            #print(reward)
        return reward, done, distance

    def find_in_list_of_list(self, mylist, char):
        for sub_list in mylist:
            if char in sub_list:
                return (mylist.index(sub_list), sub_list.index(char))

    def does_list_exist(self, mylist, char):
        exist = False
        for sub_list in mylist:
            if char in sub_list:
                exist = True
                return exist
        return exist

    def add_goal(self):

        if self.oval != None:
            self.canvas.delete(self.oval)
        #self.goalpos = np.random.randint(len(self.goals))
        self.goalpos=self.next_goal_index
        self.next_goal_index=(self.next_goal_index+1)%len(self.goals) # make it cycle round the goals in a loop
        vector = Vector2D(self.goals[self.goalpos][1][0], self.goals[self.goalpos][1][1])
        self.oval = self.canvas.create_oval(
            vector.x - 15, vector.y - 15,
            vector.x + 15, vector.y + 15,
            fill='yellow')
        # index = self.find_in_list_of_list(self.goals,vector.convert_to_numbers(self.units))
        # self.goals[index[0]][index[1]] = self.oval

    def add_hell(self, vector):
        vector_ = [vector.x, vector.y]
        hell = self.canvas.create_rectangle(
            vector_[0] - 15, vector_[1] - 15, vector_[0] + 15, vector_[1] + 15, fill='black'
        )
        index = self.find_in_list_of_list(self.hells, vector.convert_to_numbers(self.units))
        self.hells[index[0]][0] = hell

    def add_agent(self):
        for agent in self.agents:
            vector_ = agent[1]
            self.rect = self.canvas.create_rectangle(vector_[0] - 15, vector_[1] - 15, vector_[0] + 15, vector_[1] + 15,
                                            fill='red')
        #index = self.find_in_list_of_list(self.agents, vector_)
        #self.agents[index[0]][index[1]] = self.rect

    def render(self):
        if Graphics:
            self.update()




    def update_qtable(self,numbers, q_value, nq_value = None):
        if len(nq_value) != 0:
            nq_value = list(nq_value)
            nqmax = [i for i, x in enumerate(nq_value) if x == max(nq_value)]
            if len(nqmax) == 4:
                nanswer = 'equal'
            else:
                nanswer = str([self.action_space[index] for index in nqmax])

        q_value = list(q_value)
        qmax =   [i for i, x in enumerate(q_value) if x == max(q_value)]
        if len(qmax) == 4:
            answer = 'equal'
        else:
            answer = str([self.action_space[index] for index in qmax])

        if (len(q_value) == 5):


            if len(nq_value) != 0:

                text = "    {head1:4s}  {head2:4s}" \
               "\n -----------------------" \
               "\n{sup:4s}    {up:.4f}  {nup:.4f}" \
               "\n{sdown:4s}    {down:.4f}  {ndown:.4f}" \
               "\n{sright:4s}   {right:.7f} {nright:.7f}" \
               "\n{sleft:4s}    {left:.7f}  {nleft:.7f}" \
               "\n{sstop:4s}    {stop: .4f} {nstop:.4f}" \
                "\n highest     {h:4s} {nh:4s}"
                self.canvas.itemconfig(self.q_table[str(numbers[0]) + str(numbers[1])],
                                       text=str(text.format(up=q_value[0],
                                                            down=q_value[
                                                                1],
                                                            right=q_value[
                                                                2],
                                                            left=q_value[
                                                                3],
                                                            stop=q_value[
                                                                4],
                                                            nup=nq_value[
                                                                0],
                                                            ndown=nq_value[
                                                                1],
                                                            nright=
                                                            nq_value[2],
                                                            nleft=nq_value[
                                                                3],
                                                            nstop=nq_value[
                                                                4],
                                                            head1="Visited",
                                                            head2="not_Visited",
                                                            sup="up",
                                                            sdown="down",
                                                            sright="right",
                                                            sleft="left",
                                                            sstop="stop",
                                                            h=answer,
                                                            nh=nanswer
                                                            )))
            else:
                text = "    {head1:4s}" \
                       "\n -----------------------" \
                       "\n{sup:4s}    {up:.4f}  " \
                       "\n{sdown:4s}    {down:.4f}" \
                       "\n{sright:4s}   {right:.7f}" \
                       "\n{sleft:4s}    {left:.7f} " \
                       "\n{sstop:4s}    {stop: .4f}" \
                       "\n highest     {h:4s} {nh:4s}"
                #numbers = [int(s) for s in [n.strip() for n in numbers] if s.isdigit()]
                self.canvas.itemconfig(self.q_table[str(numbers[0]) + str(numbers[1])],
                                       text=str(text.format(up=q_value[0],
                                                            down=q_value[
                                                                1],
                                                            right=q_value[
                                                                2],
                                                            left=q_value[
                                                                3],
                                                            stop=q_value[
                                                                4],

                                                            head1="Visited",

                                                            sup="up",
                                                            sdown="down",
                                                            sright="right",
                                                            sleft="left",
                                                            sstop="stop",
                                                            h=answer

                                                            )))

        else:
            if len(nq_value) != 0:
                text = "    {head1:4s}  {head2:4s}" \
                   "\n -----------------------" \
                   "\n{sup:4s}    {up:.4f}  {nup:.4f}" \
                   "\n{sdown:4s}    {down:.4f}  {ndown:.4f}" \
                   "\n{sright:4s}   {right:.7f} {nright:.7f}" \
                   "\n{sleft:4s}    {left:.7f}  {nleft:.7f}" \
                    "\n highest     {h:5s}       {nh:5s}"


                self.canvas.itemconfig(self.q_table[str(numbers[0]) + str(numbers[1])],
                                           text=str(text.format(up=q_value[0],
                                                                down=q_value[1],
                                                                right=q_value[2],
                                                                left=q_value[3],

                                                                nup=nq_value[0],
                                                                ndown=nq_value[1],
                                                                nright=nq_value[2],
                                                                nleft=nq_value[3],

                                                                head1="Visited",
                                                                head2="not_Visited",
                                                                sup="up",
                                                                sdown="down",
                                                                sright="right",
                                                                sleft="left",
                                                                h=answer,
                                                                nh=nanswer
                                                                )))

            else:

                text = "    {head1:4s}" \
                       "\n -----------------------" \
                       "\n{sup:4s}    {up:.4f}  " \
                       "\n{sdown:4s}    {down:.4f}" \
                       "\n{sright:4s}   {right:.7f}" \
                       "\n{sleft:4s}    {left:.7f} " \
                       "\n highest     {h:4s} "


                #numbers = [int(s) for s in [n.strip() for n in numbers] if s.isdigit()]


                self.canvas.itemconfig(self.q_table[str(numbers[0]) + str(numbers[1])],
                                       text=str(text.format(up=q_value[0],
                                                            down=q_value[
                                                                1],
                                                            right=q_value[
                                                                2],
                                                            left=q_value[
                                                                3],
                                                            head1="Visited",
                                                            sup="up",
                                                            sdown="down",
                                                            sright="right",
                                                            sleft="left",
                                                            h=answer

                                                            )))

'''
Main LOOP
'''
def update():

    #ATTENTION: every iteration first runs with ep and learning and then runs again on no ep and no learning therefore, t_w_e is with epsilon and t_w_ is without epsilon
    # e indicates if epsilon and learning was used.
    # The CSV file will save ['iteration','map_name','algorithm_name','Algorithm_has_memory','TabularQ','reward_no_e',
    #                          'steps_no_e', 'finished_in_goal_no_e', 'finished_in_hell_no_e', 'ran_out_of_steps_no_e',
    #                          'reward_with_e', 'steps_with_e', 'finished_in_goal_with_e', 'finished_in_hell_with_e',
    #                          'ran_out_of_steps_with_e', 'goal_position', 'reward_decay', 'max_steps'])
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
            if brain == 2 or brain ==3:
                RL.reset_memory()
            while True:
                env.render()
                current_state = Vector2D.convert_to_numbers_raw(state, env.units)
                if brain == 2:
                    current_state.append(int(RL.Memory[current_state[0], current_state[1]]))
                if  brain == 3:
                    #print(current_state[0], current_state[1],action,RL.Memory[current_state[0], current_state[1],action])
                    current_state.append(int(RL.Memory[current_state[0], current_state[1],action]))
                action = RL.choose(current_state, True )
                # RL take action and get next observation and reward
                state_, reward, done, coords, steps, distance, f_h, f_g, r_s, goal_pos = env.step(action)

                step_ = steps
                next_state = Vector2D.convert_to_numbers_raw(state_, env.units)
                if brain == 2:
                    next_state.append(int(RL.Memory[next_state[0], next_state[1]]))
                if brain == 3:
                    next_state.append(int(RL.Memory[next_state[0], next_state[1],action]))

                # RL learn from this transition
                next_action = RL.choose(next_state, True)
                q_value, s = RL.learn(current_state, action, reward, next_state,next_action, done)
                if brain == 2 or brain == 3:
                    s = [s[0], s[1]]
                #if current_state == [3, 1, 0] or current_state == [3,1,1]:
                #    print(current_state, q_value)

                if brain == 2 or brain == 3:

                    env.update_qtable(s, q_value[0], q_value[1])
                else:
                    print(">>>",q_value)
                    print("??>?", s)
                    env.update_qtable(s,q_value,[])
                state = state_
                prev_action = action
                t_w_e += reward
                if (done):
                    f_h_e = f_h
                    f_g_e = f_g
                    r_s_e = r_s
                    break
        print("Finished episodes, t_w_e", t_w_e) 
        for episode in range(episodes):
            # initial observation
            state, goal_pos = env.reset()
            env.render()
            if brain == 3 or brain ==4:
                RL.reset_memory()
            while True:
                env.render()
                current_state = Vector2D.convert_to_numbers_raw(state, env.units)

                if brain == 2:
                    current_state.append(int(RL.Memory[current_state[0], current_state[1]]))
                if brain == 3:
                    current_state.append(int(RL.Memory[current_state[0], current_state[1],action]))
                action = RL.choose(current_state, False )

                # RL take action and get next observation and reward
                state_, reward, done, coords, steps, distance, f_h, f_g, r_s, goal_pos = env.step(action)
                step_e = steps
                next_state = Vector2D.convert_to_numbers_raw(state_, env.units)
                if brain == 2:
                    next_state.append(int(RL.Memory[next_state[0], next_state[1]]))
                if brain ==3:
                    next_state.append(int(RL.Memory[next_state[0], next_state[1],action]))

                # RL learn from this transition
                #next_action = RL.choose(next_state, False)
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
        print("Finished episodes, t_w", t_w_) 
        data.append([iteration, current_map, current_AI, is_memory, TabularQ, t_w_/episodes, step_, f_g_, f_h_, r_s_ , t_w_e/ episodes,step_e, f_g_e, f_h_e, r_s_e, goal_pos, RL.reward_decay,max_steps])
        #if iteration % 100 == 0:
            #print("iteraton", iteration,"brain", brain, "current_map", current_map, t_w)
        #print("Average reward for iteration:",iteration, " is ",t_w/episodes)
    env._snapsaveCanvas(logpath + "Tablememv1" + ".png")
    RL.Tab.to_csv(current_map+"_"+current_AI+".csv")
    env.destroy()




if __name__ == "__main__":


    '''Iteration,map_name,algorithm_name,Algorithm_has_memory,TabularQ, reward'''



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
    Memory_size = 2
    episodes = 4
    path = os.path.dirname(__file__).split("/")
#    print("path",path,os.path.dirname(__file__))
    mazepath = path[:]
    logpath = path[:]
    logpath.append('logs')
    mazepath.append('maps')
    mazepath = "."+''.join([str(elem) + "/" for elem in mazepath])
    logpath = "."+''.join([str(elem) + "/" for elem in logpath])
    for i in range(1):

        print("TRIAL ", i)
        for brain in brains:
            for map_index in range(len(maps_name)):
                current_map = maps_name[map_index]
                max_steps = steps[map_index]
                if brain == 1:
                    env = dynamic_maze( maps_name[map_index], max_steps, False)
                    RL = TabularQLearningWithoutMemory([env.maze_w, env.maze_h], actions=list(range(env.n_actions)),is_offpolicy=True)
                    current_AI = "OFFPolicyTabularQNoMemory"
                    is_memory = False
                    TabularQ = True
                    env.after(100, update)
                    env.mainloop()
                elif brain ==2:
                    env = dynamic_maze( maps_name[map_index], max_steps, False)
                    RL = TabularQLearningWithMemoryV1([env.maze_w, env.maze_h,Memory_size], actions=list(range(env.n_actions)),
                                                       is_offpolicy=True)
                    current_AI = "OFFPolicyTabularQMemoryV1"
                    is_memory = True
                    TabularQ = True
                    env.after(100, update)
                    env.mainloop()
                elif brain ==3:
                    env = dynamic_maze( maps_name[map_index], max_steps, False)
                    RL = TabularQLearningWithMemoryV2([env.maze_w, env.maze_h, Memory_size], actions=list(range(env.n_actions)),
                                                       is_offpolicy=True)
                    current_AI = "OFFPolicyTabularQMemoryV2"
                    is_memory = True
                    TabularQ = True
                    env.after(100, update)
                    env.mainloop()

        #rows=  zip(data[0], data[1], data[2],data[3],data[4],data[5],data[6],data[7],data[8])
        print(np.shape(data))
        with open("experiment_results_trials_e_change_"+str(i)+".csv", "w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['iteration','map_name','algorithm_name','Algorithm_has_memory','TabularQ','reward_no_e', 'steps_no_e', 'finished_in_goal_no_e', 'finished_in_hell_no_e', 'ran_out_of_steps_no_e','reward_with_e', 'steps_with_e', 'finished_in_goal_with_e', 'finished_in_hell_with_e', 'ran_out_of_steps_with_e', 'goal_position', 'reward_decay', 'max_steps'])
            for row in data:
                writer.writerow(row)

        data = []
        f.close()





