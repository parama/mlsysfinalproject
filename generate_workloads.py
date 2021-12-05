import argparse
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/33331087/sampling-from-a-bounded-domain-zipf-distribution
def main(cmd_args):
    dataset_path = cmd_args.dataset
    output_path = cmd_args.output
    dtype_raw = cmd_args.dtype
    if dtype_raw == "unit64":
        dtype = np.uint64
    alpha = cmd_args.alpha
    workload_size = cmd_args.size
    visualize = cmd_args.visualize

    assert alpha > 1, "alpha should be greater than 1, current value is {}".format(alpha)
    assert os.path.isfile(dataset_path), "{} does not exist".format(dataset_path)

    dataset = np.fromfile(dataset_path, dtype=dtype)[1:]

    # construct frequency dictionary
    freq_dict = dict()
    for x in tqdm.tqdm(dataset, desc="parsing keys"):
        freq_dict[x] = freq_dict.get(x, 0) + 1

    # compute ways
    print("Constructing Zipf distribution")
    samples = np.array(sorted(list(freq_dict.keys()), key=lambda num: freq_dict[num], reverse=True))
    weights = np.linspace(1, samples.shape[0], samples.shape[0])
    weights = np.power(weights, alpha)
    weights = np.divide(1, weights)
    weights = np.divide(weights, np.sum(weights))
    workload = np.random.choice(samples, p=weights, size=workload_size, replace=True)

    # save workloads
    output_filename = os.path.join(output_path, os.path.basename(dataset_path) + "_workload{}k_alpha{}".format(int(workload_size/1000), alpha))
    workload.astype('int64').tofile(output_filename)

    # show a preview of generated workloads
    if visualize:
        plt.hist(workload, log=True, bins=50)
        plt.xlabel("Key", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.title("Histogram of generated workload", fontsize=15)
        plt.show()


if __name__ == "__main__":

    # example usage:
    # python3 generate_workloads.py -d SOSD/data/wiki_ts_200M_uint64 -t unit64 -s 100000 -o workloads -a 2

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help="path to input data")
    parser.add_argument('-t', '--dtype', type=str, default="unit64", choices=["unit64"], help="input data format")
    parser.add_argument('-s', '--size', type=int, help="size of workload")
    parser.add_argument('-o', '--output', type=str, help="path to output workload")
    parser.add_argument('-a', '--alpha', type=float, default=2, help="parameter for Zipf distribution")
    parser.add_argument('-v', '--visualize', action='store_true', help="show preview of generated workload")
    args = parser.parse_args()
    main(args)
