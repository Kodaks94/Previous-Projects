import numpy as np
import tkinter as tk
import tkinter.font as tkf
from Utility.Vector2D import Vector2D

from PIL import ImageGrab


class dynamic_maze(tk.Tk, object):

    def __init__(self, maze_name, maxsteps, is_deep):
        try:
            if maze_name[-4] != ".npy":
                maze_name += ".npy"
            data = np.load( maze_name, allow_pickle=True)
        except IOError as e:
            raise


        spec, self.hells, self.agents, self.goals, self.labelled_dict_list = data
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
        self.goalpos = np.random.randint(len(self.goals))
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

        self.update()




    def update_qtable(self,numbers, q_value, nq_value = None):



        if nq_value.all() != None:
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


            if nq_value != None:

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
                numbers = [int(s) for s in [n.strip() for n in numbers] if s.isdigit()]
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
            if nq_value != None:
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


                numbers = [int(s) for s in [n.strip() for n in numbers] if s.isdigit()]

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
