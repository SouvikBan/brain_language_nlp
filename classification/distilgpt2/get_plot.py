import numpy as np
import seaborn as sns
import pandas as pd
import argparse

def get_accuracy(filename):
    x = np.load(filename, allow_pickle=True)
    x = x.mean(0)

    delta = 1e-3
    inc = 1e-3
    fdp = 1

    while fdp >= .05 and delta < 0.5:
        fdp = (1 + x[np.where(x <= 0.5 - delta)].shape[0]) / (1 + x[np.where(x >= 0.5 + delta)].shape[0])
        delta += inc

    return x[np.where(x >= 0.5 + delta)].shape[0] / x.shape[0]

def generate_plot(model_name, subjects, layers, outfile):
    data = list()
    for s in subjects:
        for l in layers:
            #data.append([s, l, "1", 100 * get_accuracy("1_" + s + "_" + l + "_accs.pkl")])
            data.append([s, l, "10", 100 * get_accuracy("10_" + s + "_" + l + "_accs.pkl")])
    df = pd.DataFrame(data=data, columns=["Subject", "Layer", "Context Length", "Accuracy"])
    g = sns.catplot(x="Layer", y="Accuracy", col="Subject", col_wrap=2, hue="Context Length", data=df, kind="bar")
    g.fig.suptitle(model_name)
    g.savefig(outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, nargs='+')
    parser.add_argument("--layers", type=str, nargs='+')
    parser.add_argument("--model", type=str)
    parser.add_argument("--save_file", type=str)

    args = parser.parse_args()
    generate_plot(args.model, args.subjects, args.layers, args.save_file)
