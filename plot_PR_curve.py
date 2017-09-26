import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def plot_pr_curve(data_csv, size, name):
    plt.switch_backend('agg')
    df = pd.read_csv(data_csv)
    legends = ['all', 'small', 'large']
    
    plt.figure(figsize=size)
    for legend in legends:
        y_real = (np.array(df[df['size'].isin(['small', 'large'] if legend=='all' else [legend])]['real']) > 0).astype(int)
        y_scores = np.array(df[df['size'].isin(['small', 'large'] if legend=='all' else [legend])]['scores'])
        precision, recall, _ = precision_recall_curve(y_real, y_scores)
        plt.plot(recall, precision, label=legend + ' : AUC={0:0.2f}'.format(average_precision_score(y_real, y_scores)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for {}'.format(name))
    plt.legend(loc="upper right")
    plt.savefig(data_csv.split('/')[-1].split('.')[0] + '.png')
    
    
if __name__ == '__main__':
    data = sys.argv[1]
    size = (int(sys.argv[2]), int(sys.argv[3]))
    name = sys.argv[4]
    plot_pr_curve(data, size, name)