'''
Program Created by Mahrad Pisheh Var
@ University of Essex

'''

import numpy as np
import random
import string
import csv
import os
import pygame

class maze_cell(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        # TOP DOWN Left Right
        self.wall_description = {'up':True,'down': True, 'left':True, 'right': True}
        self.neighbour_direction = {'up':(-1, 0),'down': (1,0), 'left':(0, -1), 'right': (0, 1)}
        #self.neighbour_direction =  [(1, 0), (-1,0), (0, 1), (0, -1)]

    def set_visited(self):
        self.visited = True
    def is_visited(self):
        return self.visited
    def out_of_grid(self,grid, dx,dy):
        outside_grid_x = [len(grid[0]), -1]
        outside_grid_y = [-1, len(grid)]
        is_out_of_grid = self.x + dx in outside_grid_x or self.y+dy in outside_grid_y
        return is_out_of_grid
    def get_unvisited_neighbours(self, grid)->list:
        neighbours = []
        for _,dir1 in self.neighbour_direction.items():
            (dx,dy)=dir1
            if not self.out_of_grid(grid,dx,dy):
                chosen_neighbour = grid[self.y +dy][self.x +dx]
                if not chosen_neighbour.is_visited():
                    neighbours.append(chosen_neighbour)
        return neighbours
    def remove_wall(self, other):
        if other.x > self.x:
            other.wall_description['left'] = False
            self.wall_description['right'] = False
        elif other.x < self.x:
            other.wall_description['right'] = False
            self.wall_description['left'] = False
        elif other.y > self.y:
            other.wall_description['up'] = False
            self.wall_description['down'] = False
        elif other.y < self.y:
            other.wall_description['down'] = False
            self.wall_description['up'] = False
        
    def __str__(self):
        return ("(x"+str(self.x) + ",y" + str(self.y)+")")
class maze(object):
    def __init__(self, n_rows, n_columns):
        self.x = n_rows
        self.y = n_columns
        self.size = n_rows * n_columns
        self.grid = [[maze_cell(x,y) for x in range(self.x)] for y in range(self.y)]
        self.generate_dfs()
    def generate_dfs(self):
        curr_cell = self.grid[0][0]
        stack =  []
        doing_backtrack=False
        while True:
            curr_cell.set_visited()
            neighbours = curr_cell.get_unvisited_neighbours(self.grid)
            if neighbours:
                neighbour = random.choice(neighbours)
                neighbour.set_visited()
                stack.append(curr_cell)
                curr_cell.remove_wall(neighbour)
                curr_cell = neighbour
            elif stack:
                if doing_backtrack==False:
                    print("Stuck at ",curr_cell,".  Starting backtrack!")
                    doing_backtrack=True
                    #return
                curr_cell = stack.pop() 
                neighbours = curr_cell.get_unvisited_neighbours(self.grid)
                if neighbours:
                    print("Backtracking to ",curr_cell)
                    doing_backtrack=False
            else:
                break
    def convert_maze(self):
        converted_grid = []
        ylength = len(self.grid)*2+1
        xlength = len(self.grid[0])*2+1
        for y in range(ylength):
            converted_grid.append([1] * xlength)
        for y, row_of_cells in enumerate(self.grid):
            for x, cell in enumerate(row_of_cells):
                for i,direction in enumerate(cell.wall_description):
                    if cell.is_visited():
                        # they should all be visited
                        converted_grid[y*2+1][x*2+1]=0
                    if not(cell.wall_description[direction]):
                        (dy,dx)=cell.neighbour_direction[direction]
                        converted_grid[y*2+1+dy][x*2+1+dx]=0
        print('\n'.join([''.join(str(x)) for x in converted_grid]))
        return converted_grid

    def save_to_csv(self, converted_grid):
        name = input("enter the name for the maze")
        if name == "":
            name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        name += str('('+str(self.x)+ ','+ str(self.y)+')')
        with open(name+".csv", "w" ,newline="") as f:
            writer = csv.writer(f)
            for row in converted_grid:
                writer.writerow(row)

class PyGame_display():
    def __init__(self, converted_maze):
        pygame.init()
        display_cell_size=11
        green = (0,155,0)
        brown = (205,133,63)    
        display_width = len(converted_maze[0])*display_cell_size
        display_height = len(converted_maze)*display_cell_size

        pygame.display.set_caption("Maze_viewer")
        gameDisplay = pygame.display.set_mode((display_width,display_height))
        gameDisplay.fill((brown))
        for y,row_of_cells in enumerate(converted_maze):
            for x, cell in enumerate(row_of_cells):
                if cell==1:
                    pygame.draw.rect(gameDisplay, green, pygame.Rect(x*display_cell_size, y*display_cell_size, display_cell_size-1,display_cell_size-1))
        pygame.display.flip()


if __name__ == "__main__":
    #size_x = int(input("enter the width: "))
    #size_y = int(input("enter the height: "))
    size_x=10
    size_y=10
    #m = maze(size_x, size_y)
    #converted = m.convert_maze()
    m = np.genfromtxt("complex(10,10).csv", delimiter=',')
    PyGame_display(m)
    input("Press enter")
    #m.save_to_csv(converted)
