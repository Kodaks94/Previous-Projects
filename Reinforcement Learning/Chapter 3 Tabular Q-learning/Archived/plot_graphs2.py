import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys
data_trials = []
a = 2000

df = pd.read_csv("experiment_results_trials_e_change_0.csv")
df['reward_wth_e_smoothed'] = 0
df['reward_no_e_smoothed'] = 0
df['steps_with_e_smoothed'] = 0
df['steps_no_e_smoothed'] = 0
df.reward_wth_e_smoothed = df['reward_with_e'].rolling(a).mean()
df.reward_no_e_smoothed = df['reward_no_e'].rolling(a).mean()
df.steps_with_e_smoothed = df['steps_with_e'].rolling(a).mean()
df.steps_no_e_smoothed = df['steps_no_e'].rolling(a).mean()
print(df.head())
df = df[a:]
for i in range(0,1):
    print("experiment_results_trials_"+str(i)+".csv")
    df2 = pd.read_csv("experiment_results_trials_e_change_"+str(i)+".csv")
    df2['reward_wth_e_smoothed'] = 0
    df2['reward_no_e_smoothed'] = 0
    df2['steps_with_e_smoothed'] = 0
    df2['steps_no_e_smoothed'] = 0
    df2.reward_wth_e_smoothed = df['reward_with_e'].rolling(a).mean()
    df2.reward_no_e_smoothed = df['reward_no_e'].rolling(a).mean()
    df2.steps_with_e_smoothed = df['steps_with_e'].rolling(a).mean()
    df2.steps_no_e_smoothed = df['steps_no_e'].rolling(a).mean()
    df2 = df2[a:]
    df = df.append(df2, ignore_index = True)
    print(len(df), len(df2))
    df.head()
map_names = df["map_name"].unique()
#print(df.describe())
df.to_csv("temp.csv")
#sys.exit(0)

df = pd.read_csv("temp.csv")
df = df[0::a]
map_names = df["map_name"].unique()
print(map_names)
Onpolicylist = ["ONPolicyTabularQNoMemory","ONPolicyTabularQMemory", "OFFPolicyDeepQWithMemorywithReplayMemory", "OFFPolicyDeepQNoMemoryWithReplayMemory" ]
for map_name in map_names:
    df1  = df.query("map_name=='"+map_name+"'")
    df1 = df1[~df1.algorithm_name.isin(Onpolicylist)]


    sns.set_style("whitegrid")
    plt.figure(figsize=(20,9) )
    print("reached")

    #sub2 = sns.lineplot(data=df1, x="iteration", y="steps_no_e_smoothed", hue="algorithm_name", linewidth=0.4)
    #for i in range(len(sub2.lines)):
    #    sub2.lines[i].set_linestyle("--")


    sub = sns.lineplot(data=df1, x="iteration",  y="reward_with_e",  hue="algorithm_name", linewidth=0.4)


    sns.lineplot()

    #sns.lineplot(data=df1, x="iteration",  y="steps_no_e",  hue="algorithm_name", linewidth=0.4)
    print("passed")
    fig = sub.get_figure()

    sub.set(xlabel='Iteration', ylabel='Rewards')
    fig.suptitle("Effect of Algorithm on map "+map_name, fontsize=12,y=0.93)

    fig.savefig("alg_effect_for_reward_with_reward_"+map_name+".png",bbox_inches="tight")
    fig.clf()
