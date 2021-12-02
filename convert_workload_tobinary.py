import os
import argparse
import numpy as np

from collections import Counter

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def _compute_weights(data_path, workloads):
    weights_path = os.path.join(DATA_PATH, 'weights')
    os.makedirs(weights_path, exist_ok=True)
    
    data = np.fromfile(data_path, dtype=np.uint64)[1:]
    for workload in workloads:
        print("working on {}".format(workload))
        workload_data = np.fromfile(DATA_PATH + workload, dtype=np.uint64)
        counts = Counter(workload_data)
        weights = np.vectorize(counts.get)(data, 0)
        weights += 1 # for non occuring
        weights = weights.astype('float64')
        weights /= weights.max() # normalize
        print("finished with max weight {}, size {}".format(weights.max(), weights.shape))

        # save
        weights.tofile(weights_path + '/' + workload.split('/')[-1])

def convert_workload_to_weights(dataset='osmc'):
    # wiki dataset
    wiki_data_path = DATA_PATH + "/wiki_ts_200M_uint64"
    wiki_workloads = [
        "/workloads/wiki_ts_200M_uint64_workload100k_alpha1.1",
        "/workloads/wiki_ts_200M_uint64_workload100k_alpha2.0",
        "/workloads/wiki_ts_200M_uint64_workload100k_alpha3.0",
        "/workloads/wiki_ts_200M_uint64_workload100k_alpha4.0",
        "/workloads/wiki_ts_200M_uint64_workload100k_alpha5.0"
    ]

    # osmc dataset
    osmc_data_path = DATA_PATH + "/osm_cellids_200M_uint64"
    osmc_workloads = [
        "/workloads/osm_cellids_200M_uint64_workload100k_alpha1.1",
        "/workloads/osm_cellids_200M_uint64_workload100k_alpha2.0",
        "/workloads/osm_cellids_200M_uint64_workload100k_alpha3.0",
        "/workloads/osm_cellids_200M_uint64_workload100k_alpha4.0",
        "/workloads/osm_cellids_200M_uint64_workload100k_alpha5.0"
    ]
    
    if dataset == 'wiki':
        _compute_weights(wiki_data_path, wiki_workloads)
        
    elif dataset == 'osmc':
        _compute_weights(osmc_data_path, osmc_workloads)
    

def main(input_path, output_path):
    if not '.txt' in input_path:
        input_path = input_path + '.txt'

    workload = np.loadtxt(input_path)
    workload.astype('int64').tofile(output_path)

if __name__ == "__main__":

    # example usage:
    # python3 convert_workload_tobinary.py -i wiki_ts_200M_unit64_alpha2_100k -o wiki_ts_200M_uint64_workload

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="path to input workload in txt")
    parser.add_argument('-o', '--output', type=str, help="path to output workload in binary")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    main(input_path, output_path)