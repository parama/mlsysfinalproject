import argparse
import numpy as np

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