#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def main():
    save_dir = "./cole_runs/cpt_model-base/"
   
    with open(save_dir + "cole_eval_metrics.json", "r") as f_open:
        data = json.loads(f_open.read())
    print(data.keys())
    ep = EvaluationPlot(data, save_dir)
    ep.confusion_matrix()

class EvaluationPlot():
    task_labels = {
        "qfrcola": ["True", "False"],
        "qfrblimp": ["True", "False"],
    }
    def __init__(self, logs, save_dir):
        self.save_dir = save_dir
        self.logs = logs
        

    def confusion_matrix(self):
        metrics = self.logs["metrics"]
        for metric in metrics:
            task = metric["task"]
            labels = metric["labels"] if self.task_labels.get(task, None) is None else self.task_labels[task]
            fig, plot = plt.subplots(nrows=1, ncols=1)
            pdcm = pd.DataFrame(metric["confusion_matrix"], index = [i for i in labels],
                  columns = [i for i in labels])
            sn.heatmap(pdcm, annot=True, fmt='d')
            plt.title(f"Task {task} results")
            print(self.save_dir + task + ".png")
            fig.savefig(self.save_dir + task + ".png")
            plt.close(fig)



if __name__ == "__main__":
    main()
