import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2
import numpy as np
name_map = 'LongCorridor'

a = pd.read_csv(name_map+'.txt',encoding='utf-16')
#a = a.replace('iteration,averagedReward,algoname,memory','')
a = a[a.iteration != 'iteration']
#print(a)
a = a[a.iteration == '1999']
print(a)
#a = a.iloc[::1999,:]
TabularQlearning = a[a.algoname=='TabularQlearning']
DeepQLearning =  a[a.algoname=='DeepQLearning']
TabularQlearningMemory = a[a.algoname=='TabularQlearningMemory']
DeepQLearningMemory = a[a.algoname=='DeepQLearningMemory']
print(pd.to_numeric(TabularQlearning['averagedReward'],errors='coerce').mean())
print(pd.to_numeric(DeepQLearning['averagedReward'],errors='coerce').mean())
print(pd.to_numeric(TabularQlearningMemory['averagedReward'],errors='coerce').mean())
print(pd.to_numeric(DeepQLearningMemory['averagedReward'],errors='coerce').mean())
#b = a.reshape()
#print(b)



'''

a['smoothed_reward'] = 0
a.smoothed_reward = a['averagedReward'].rolling(50).mean()

algonames = a['algoname'].unique()
print(algonames)



df1 = a
sns.set_style("ticks")
plt2.figure(figsize=(20,9))

plt = sns.lineplot(data = df1, x="iteration",y= 'smoothed_reward', hue='algoname', linewidth= 0.4 )
fig = plt.get_figure()
plt.set(xlabel = 'Iteration', ylabel= 'averagedReward')
count = 0
for label in plt.xaxis.get_ticklabels():
    if count % 100 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
    count +=1

plt2.tight_layout()

#fig.suptitle("Effects of Algorithm on map "+name_map, fontsize = 12, y = 0.93)
fig.savefig("alg_effect_for_"+name_map+".png")

'''
