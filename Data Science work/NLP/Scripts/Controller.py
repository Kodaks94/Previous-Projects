import sklearn.datasets as datasets
import numpy as np
import pandas as pd
import seaborn as sb
def loadfile(filedir):

    file = pd.read_csv(filedir)
    return file

def inspectfile(file):
    print("first 10 entries")
    print(file.sample(10))
    #sb.barplot(x="temp", y="area",hue="day" ,data= file)


def main():
    inspectfile(loadfile('Datasets/forestfires.csv'))
    inspectfile(loadfile('Datasets/train.csv'))
    inspectfile(loadfile('Datasets/parkinsons.csv'))
if __name__== "__main__":
  main()



