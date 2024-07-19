# one is wall 0 is clear
import numpy as np

class Node():

    def __init__(self, parent=None, position=None):
        self.position = position
        self.parent = parent

        self.g = 0
        self.h = 0
        self.f = 0

    def reset(self):
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):

        return self.position == other.position


class Astar(object):

    def __init__(self, maze, start, end):

        self.start_node = Node(None, start)
        self.start_node.reset()
        self.end_node = Node(None, end)
        self.end_node.reset()
        self.maze = maze
        self.rows , self.columns = np.shape(maze)

    # This function return the path of the search

    def loop_to_find(self):
        self.open_list = []
        self.closed_list = []
        self.open_list.append(self.start_node)
        path = []
        outer_iteration = 0
        max_iter = (len(self.maze) //2)**10
        while len(self.open_list) > 0:
            outer_iteration +=1
            # Get the current node
            current_node = self.open_list[0]
            current_index = 0
            for index, item in enumerate(self.open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index



            self.open_list.pop(current_index)
            self.closed_list.append(current_node)

            if current_node == self.end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]  # reversed
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
                # Make sure within range
                if (node_position[0] > (self.rows - 1) or
                        node_position[0] < 0 or
                        node_position[1] > (self.columns - 1) or
                        node_position[1] < 0):
                    continue
                # Make sure walkable terrain
                if self.maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                for closed_child in self.closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - self.end_node.position[0]) ** 2) + (
                        (child.position[1] - self.end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in self.open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                self.open_list.append(child)

        return path


class AstarDist(object):

    def __init__(self, maze, ncolumns, nrows, end):
        self.end = (end[1], end[0])

        self.maze = maze
        self.ncolumns = ncolumns
        self.nrows = nrows


    def loop_to_record(self):
        distance_dict = {}

        for i in range(self.nrows):
            for j in range(self.ncolumns):
                if(self.maze[i][j] == 0 ):

                    c_cord = (i,j)
                    print(c_cord)
                    distance_dict[j,i] = len(Astar(self.maze,c_cord,self.end).loop_to_find())-1



        return distance_dict
