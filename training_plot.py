#!/usr/bin/env python

import json
import matplotlib.pyplot as plt

def main():
    save_dir = "./models/cpt_model-3E/logs/"
   
    with open(save_dir + "train_log.json", "r") as f_open:
        data = json.loads(f_open.read())
    print(data.keys())
    tp = TrainingPlot(data, save_dir)
    tp.perplexity()

class TrainingPlot():
    def __init__(self, logs, save_dir):
        self.save_dir = save_dir
        self.logs = logs
        
    def perplexity(self):
        fig, plot = plt.subplots(nrows=1, ncols=1)
        plot.plot(self.logs["epoch"], self.logs["perplexity"])
        plt.ylabel("Perplexity")
        plt.xlabel("Epoch")
        print(self.save_dir + "perplexity.png")
        fig.savefig(self.save_dir + "perplexity.png")
        plt.close(fig)



if __name__ == "__main__":
    main()
