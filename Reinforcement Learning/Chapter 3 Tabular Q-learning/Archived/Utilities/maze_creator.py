import numpy as np
import tkinter as tk
from PIL import ImageGrab
from  Utility.Vector2D import Vector2D
from  Brains.Astar import AstarDist
import os
import uuid

class maze_creator(tk.Tk, object):
    def __init__(self, maze_w, maze_h, units):
        self.units = units
        self.maze_h = maze_h
        self.maze_w = maze_w
        self.width = maze_w * units
        self.height = maze_h * units
        super(maze_creator, self).__init__()
        self.title = "maze_maker"
        self.geometry('{0}x{1}'.format(self.width, self.height))
        self.build()
        self.hells = []
        self.agent_coords = []
        self.goal_coords = []

        self.quit = False
        scroll_bar = tk.Scrollbar(self, orient = tk.VERTICAL, command = self.canvas.yview)
        scrollable_frame = tk.Frame(self.canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scroll_bar.set)
        scroll_bar.pack(side="right", fill="y")
        self.outline = 5


    def does_list_exist(self, mylist, char):
        exist = False
        for sub_list in mylist:
            if char in sub_list:
                exist = True
                return exist
        return exist
    def build_Astar_label(self):
        labelled_maze = []
        for i in range(self.maze_h):
            row_vals = []
            for j in range(self.maze_w):
                label = lambda : 0 if ( not self.does_list_exist(self.hells,[j,i])) else 1
                row_vals.append(label())
            labelled_maze.append(row_vals)
        return labelled_maze
    def find_in_list_of_list(self,mylist, char):
        for sub_list in mylist:
            if char in sub_list:
                return (mylist.index(sub_list), sub_list.index(char))

    def check_cell(self, vector):
        if vector.convert_to_numbers(self.units) in [j for i in self.hells for j in i]:
            index_ = self.find_in_list_of_list(self.hells, vector.convert_to_numbers(self.units))
            print(index_)
            self.canvas.delete(self.hells[index_[0]][0])
            self.hells.remove(self.hells[index_[0]])
            print("removed ", vector.convert_to_numbers(self.units),"from hell list, current size: ",len(self.hells))
            return True
        elif vector.convert_to_numbers(self.units) in [j for i in self.agent_coords for j in i]:
            index_ = self.find_in_list_of_list(self.agent_coords, vector.convert_to_numbers(self.units))
            print(index_)
            self.canvas.delete(self.agent_coords[index_[0]][0])
            self.agent_coords.remove(self.agent_coords[index_[0]])
            print("removed ", vector.convert_to_numbers(self.units), "from agent list, current size: ", len(self.agent_coords))
            return True
        elif vector.convert_to_numbers(self.units) in [j for i in self.goal_coords for j in i]:
            index_ = self.find_in_list_of_list(self.goal_coords, vector.convert_to_numbers(self.units))
            print(index_)
            self.canvas.delete(self.goal_coords[index_[0]][0])
            self.goal_coords.remove(self.goal_coords[index_[0]])
            print("removed ", vector.convert_to_numbers(self.units), "from goal list, current size: ", len(self.goal_coords))
            return True

    def add_goal(self, vector):
        if not self.check_cell(vector):
            vector_ = vector.get_center_coords(self.units)
            #oval = self.canvas.create_oval(
            #    vector_[0] - 15, vector_[1] - 15,
            #    vector_[0] + 15, vector_[1] + 15,
            #    fill='yellow')
            oval = self.canvas.create_oval(
                vector_[0] - self.units / 2 + self.outline, vector_[1] - self.units / 2+ self.outline, vector_[0] + self.units / 2- self.outline,
                vector_[1] + self.units / 2- self.outline, fill='yellow'
            )
            self.canvas.create_text(self.canvas.coords(oval)[0]+self.units/2- self.outline , self.canvas.coords(oval)[1]+self.units/2- self.outline, text="EXIT "+ str(len(self.goal_coords)+1), font=("Times New Roman", 18, "bold"))

            self.goal_coords.append([oval, vector_,vector.convert_to_numbers(self.units)])
    def _canvas(self):

        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        box = (x, y, x1, y1)
        return box
    def _snapsaveCanvas(self, filename):
        print(filename)
        self.render()
        canvas = self._canvas()  # Get Window Coordinates of Canvas
        self.grabcanvas = ImageGrab.grab(bbox=canvas).save(str(filename)+".jpg")

    def add_hell(self,vector):
        if not self.check_cell(vector):
            vector_ = vector.get_center_coords(self.units)
            hell = self.canvas.create_rectangle(
                vector_[0] - self.units/2, vector_[1]- self.units/2, vector_[0] + self.units/2, vector_[1] + self.units/2, fill = 'white', outline= 'black', width= 5
            )
            self.hells.append([hell, vector_,vector.convert_to_numbers(self.units)])

    def add_agent(self, vector):
        if not self.check_cell(vector):
            vector_ = vector.get_center_coords(self.units)
            #rect = self.canvas.create_rectangle(vector_[0] , vector_[1], vector_[0] + 15, vector_[1] + 15, fill = 'red')
            rect = self.canvas.create_rectangle(
                vector_[0] - self.units/2 + self.outline, vector_[1]- self.units/2+ self.outline, vector_[0] + self.units/2- self.outline, vector_[1] + self.units/2- self.outline, fill = 'red'
            )
            self.canvas.create_text(self.canvas.coords(rect)[0] + self.units/2- self.outline, self.canvas.coords(rect)[1] + self.units/2- self.outline,
                                    text="START", font=("Times New Roman", 18, "bold"))
            self.agent_coords.append([rect,vector_,vector.convert_to_numbers(self.units)])

    def callback_goal(self, event):
        coords = Vector2D(event.x,event.y)
        self.add_goal(coords)
        print("added_goal", coords.convert_to_numbers(self.units))

    def callback_hell(self,event):
        coords = Vector2D(event.x,event.y)
        self.add_hell(coords)
        print("added_hell", coords.convert_to_numbers(self.units))

    def callback_agent(self,event):
        coords = Vector2D(event.x, event.y)
        self.add_agent(coords)
        print("added_agent", coords.convert_to_numbers(self.units))


    def Astar_init(self):
        self.labelled_dict_list = []
        self.labelled_maze = self.build_Astar_label()
        print(self.labelled_maze)
        for i in self.goal_coords:
            self.labelled_dict_list.append(AstarDist(self.labelled_maze,self.maze_w,self.maze_h, i[2]).loop_to_record())

        print(self.labelled_dict_list)

    def callback_save(self, event):
        print(repr(event.char))
        if event.char == 's':
            print("wait until Astar calculates...")
            self.Astar_init()
            print("Done.. proceed//")
            name = input("enter the name of the maze >>")
            map_log = [[self.maze_w, self.maze_h, self.units],self.hells,self.agent_coords,self.goal_coords, self.labelled_dict_list]
            numpy_conv = np.array(map_log)
            np.save(path+name, numpy_conv)
            print("Map is saved in maps/"+name+".npz")
            self.quit = True
        if event.char == 'c':
            print("screenshot_saved")
            self._snapsaveCanvas(str(uuid.uuid4()))

    def build(self):

        self.canvas = tk.Canvas(self, height = self.height, width= self.width, bg= 'white')
        self.canvas.focus_set()
        for c in range(0, self.width, self.units):
            line = c, 0, c, self.height
            #self.canvas.create_line(line[0], line[1], line[2], line[3])
        for r in range(0, self.height, self.units):
            line = 0, r, self.width, r
            #self.canvas.create_line(line[0], line[1], line[2], line[3])
        self.canvas.bind("<Button-1>", self.callback_hell)
        self.canvas.bind("<Button-2>", self.callback_agent)
        self.canvas.bind("<Button-3>", self.callback_goal)
        self.canvas.bind("<Key>", self.callback_save)
        self.canvas.pack()


    def render(self):
        self.update()
        if(self.quit == True):
            return False
        else:
            return True

    def convert_to_coords(self, n):
        unit = self.units / 2
        column = n[0]
        row = n[1]
        row_coords = row * self.units + unit
        column_coords = column * self.units + unit
        return [column_coords, row_coords, column_coords + 30, row_coords + 30]

    def convert_to_numbers(self, s):
        unit = self.units / 2
        if not self.is_deep:
            if isinstance(s, str):
                s = s.replace("  ", " ")
                s = s[1:-1].split(",")
                s = list(filter(lambda a: a != "", s))
        row = float(s[1])
        column = float(s[0])
        # n = row-5 / 40
        row_number = (row - unit) / self.units
        column_number = (column - unit) / self.units
        return [int(column_number), int(row_number)]


if __name__ == "__main__":
    newmaze = maze_creator(7,7,100)
    mappath = os.path.dirname(__file__).split("/")
    mappath = mappath[:-1]
    mappath.append('maps')
    path = ''.join([str(elem)+"/" for elem in mappath])
    print(path)
    while True:
        newmaze.render()
        if newmaze.quit == True:
            print("done shutting down")

            break

    newmaze.destroy()
