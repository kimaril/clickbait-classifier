import numpy as np
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

def plot_cm(cm, classes, classifier_name="", size=(10,7)):
    df_cm = DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=size)
    plt.title("{} Confusion Matrix".format(classifier_name))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
