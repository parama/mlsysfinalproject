import os
import numpy as np
import subprocess


def get_cmd_results(cmd):
    return subprocess.check_output(cmd, shell=True).decode('utf-8').rstrip("\n").split("\t")


def eval_wiki():
    wiki_data_path = "SOSD/data/wiki_ts_200M_uint64"
    wiki_workloads = [
        "workloads/wiki_ts_200M_uint64_workload200000k_alpha1.1",
        #"workloads/wiki_ts_200M_uint64_workload200000k_alpha1.3",
        "workloads/wiki_ts_200M_uint64_workload200000k_alpha1.5",
        #"workloads/wiki_ts_200M_uint64_workload200000k_alpha1.7",
        "workloads/wiki_ts_200M_uint64_workload200000k_alpha1.9"
    ]

    num_records = 200000000
    workload_size = 200000000

    #num_second_level_models = [10, 50, 100, 200, 500, 1000, 2000]
    num_second_level_models = [10, 50, 100, 500, 1000]
    #lookup_table_sizes = [3, 5, 10]
    # lookup_table_sizes = [100, 200, 300]
    lookup_table_sizes = [500]

    # models = ["linear_model", "weighted_linear_model", "look_up_table_linear_model"]
    models = ["look_up_table_linear_model"]

    log_path = "results/wiki_200000k_v2.csv"
    if os.path.isfile(log_path):
        log = open(log_path, "a")
    else:
        log = open(log_path, "w")
        log.write("model,num_second_level_models,table_size,train_workload,test_workload,model_size,build_time,test_workload_time,num_last_mile_search\n")

    for workload in wiki_workloads:
    
        for nslm in num_second_level_models:

            for idx, model in enumerate(models):

                if model == "linear_model":  # linear model
                    cmd = "./build/benchmark_learned_index {} {} {} {} {}".format(
                        nslm, wiki_data_path, workload, num_records, workload_size
                    )
                    msize, btime, wtime, nlms = get_cmd_results(cmd)
                    info = "{},{},{},{},{},{},{},{},{}".format(
                        model, nslm, "NA", "NA", workload, msize, btime, wtime, nlms
                    )
                    print(info)
                    log.write(info + "\n")

                weight_path = os.path.join("weights", os.path.basename(workload))
                if model == "weighted_linear_model":  # weighted linear model
                    cmd = "./build/benchmark_weighted_learned_index {} {} {} {} {} {}".format(
                        nslm, wiki_data_path, weight_path, workload, num_records, workload_size
                    )
                    msize, btime, wtime, nlms = get_cmd_results(cmd)
                    info = "{},{},{},{},{},{},{},{},{}".format(
                        model, nslm, "NA", workload, workload, msize, btime, wtime, nlms
                    )
                    print(info)
                    log.write(info + "\n")

                if model == "look_up_table_linear_model":   # lookup table linear model
                    for tsize in lookup_table_sizes:
                        cmd = "./build/benchmark_look_up_table_learned_index {} {} {} {} {} {} {}".format(
                            nslm, tsize, wiki_data_path, weight_path, workload, num_records, workload_size
                        )
                        msize, btime, wtime, nlms = get_cmd_results(cmd)
                        info = "{},{},{},{},{},{},{},{},{}".format(
                            model, nslm, tsize, workload, workload, msize, btime, wtime,nlms
                        )
                        print(info)
                        log.write(info + "\n")


    log.close()

def eval_book():
    book_data_path = "SOSD/data/books_200M_uint64"
    book_workloads = [
        "workloads/books_200M_uint64_workload200000k_alpha1.1",
        "workloads/books_200M_uint64_workload200000k_alpha1.5",
        "workloads/books_200M_uint64_workload200000k_alpha1.9"
    ]

    num_records = 200000000
    workload_size = 200000000

    num_second_level_models = [10, 50, 100, 500, 1000]
    lookup_table_sizes = [200, 500]

    # models = ["linear_model", "weighted_linear_model", "look_up_table_linear_model"]
    models = ["look_up_table_linear_model"]

    log_path = "results/books_200000k_v2.csv"
    if os.path.isfile(log_path):
        log = open(log_path, "a")
    else:
        log = open(log_path, "w")
        log.write("model,num_second_level_models,table_size,train_workload,test_workload,model_size,build_time,test_workload_time,num_last_mile_search\n")

    for workload in book_workloads:
    
        for nslm in num_second_level_models:

            for idx, model in enumerate(models):

                if model == "linear_model":  # linear model
                    cmd = "./build/benchmark_learned_index {} {} {} {} {}".format(
                        nslm, book_data_path, workload, num_records, workload_size
                    )
                    msize, btime, wtime, nlms = get_cmd_results(cmd)
                    info = "{},{},{},{},{},{},{},{},{}".format(
                        model, nslm, "NA", "NA", workload, msize, btime, wtime, nlms
                    )
                    print(info)
                    log.write(info + "\n")

                if model == "weighted_linear_model":  # weighted linear model
                    weight_path = os.path.join("weights", os.path.basename(workload))
                    cmd = "./build/benchmark_weighted_learned_index {} {} {} {} {} {}".format(
                        nslm, book_data_path, weight_path, workload, num_records, workload_size
                    )
                    msize, btime, wtime, nlms = get_cmd_results(cmd)
                    info = "{},{},{},{},{},{},{},{},{}".format(
                        model, nslm, "NA", workload, workload, msize, btime, wtime, nlms
                    )
                    print(info)
                    log.write(info + "\n")

                if model == "look_up_table_linear_model":   # lookup table linear model
                    for tsize in lookup_table_sizes:
                        cmd = "./build/benchmark_look_up_table_learned_index {} {} {} {} {} {} {} {}".format(
                            nslm, tsize, book_data_path, workload, workload, num_records, workload_size, workload_size
                        )
                        msize, btime, wtime, nlms = get_cmd_results(cmd)
                        info = "{},{},{},{},{},{},{},{},{}".format(
                            model, nslm, tsize, workload, workload, msize, btime, wtime,nlms
                        )
                        print(info)
                        log.write(info + "\n")


    log.close()


if __name__ == "__main__":
    eval_book()

