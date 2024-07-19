
# coding: utf-8

# # Analyser for Stimulies
# ## (Carries the real time timestamps and current FPS and plots them)

# In[5]:



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D


# In[12]:
'''
    CONSTANTS
'''

PROJECTION3D= False
HEATMAP = False

FILETOREAD = "Java_FrameFSP_Analyser.csv"



'''
'''


def initialize():

    if(PROJECTION3D):
        Myfig = plt.figure()
        axis_graph = plt.axes(projection='3d')
        return Myfig, axis_graph
    else:
        style.use('dark_background')

        Myfig = plt.figure()

        axis_graph = Myfig.add_subplot(1,1,1)
        return Myfig, axis_graph
def return_data():
    data  = open(FILETOREAD, 'r').read()
    entries  = data.split('\n')
    xs = []
    ys = []
    zs = []
    for entry in entries:
         if len(entry) >1:
             x , y = entry.split(',')
             xs.append(float(x))
             ys.append(float(y))
    return xs,ys
def draw(i):
    data  = open(FILETOREAD, 'r').read()
    entries  = data.split('\n')
    xs = []
    ys = []
    zs = []    
    for entry in entries:
        
        if len(entry) >1:
            if(len(entries[0].split(',')) == 3):
                x , y , z= entry.split(',')
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
            else:
                x , y= entry.split(',')
                xs.append(x)
                ys.append(y)
    axis_graph.clear()

    if(PROJECTION3D):
        axis_graph.plot(xs,ys,zs, '-b')
    else:
        axis_graph.plot(xs,ys)
    


def Draw_hm(x,y):

    heatmap , xe, ye = np.histogram2d(x,y,bins=50)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    plt.clf()
    plt.imshow(heatmap, extent = extent)
    plt.show()

    

    

Myfig, axis_graph = initialize()    
ani = animation.FuncAnimation(Myfig, draw , interval = 100 )
x,y = return_data()
#Draw_hm(x,y)
plt.show()
