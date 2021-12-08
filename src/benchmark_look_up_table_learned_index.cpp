#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#include "look_up_table_learned_index.h"

#define K uint64_t
#define V int64_t

std::vector<K> read_workload(std::string workload_path, int wl_size) {
  auto workload_data = new K[wl_size];
  std::ifstream is_workload(workload_path.c_str(), std::ios::binary | std::ios::in);
    
  is_workload.read(reinterpret_cast<char*>(workload_data),
          std::streamsize(wl_size * sizeof(K)));
  is_workload.close();

  std::vector<K> ret_workload(wl_size);
  for (int i = 0; i < wl_size; i++) {
    ret_workload[i] = workload_data[i];
  }
  return ret_workload;
}

std::vector<double> read_weights(std::string weight_path, int num_records) {
  auto weight_data = new double[num_records];
  std::ifstream is_weight(weight_path.c_str(), std::ios::binary | std::ios::in);
    
  is_weight.read(reinterpret_cast<char*>(weight_data),
          std::streamsize(num_records * sizeof(double)));
  is_weight.close();

  std::vector<double> ret_weight(num_records);
  for (int i = 0; i < num_records; i++) {
    ret_weight[i] = weight_data[i];
  }
  return ret_weight;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << "Incorrect usage." << std::endl;
    exit(1);
  }

  int num_second_level_models = atoi(argv[1]);
  int tableSize = atoi(argv[2]);
  std::string keys_file_path = std::string(argv[3]);
  std::string weights_file_path = std::string(argv[4]);
  std::string test_workload_file_path = std::string(argv[5]);
  //int num_records = 200000000;
  int num_records = atoi(argv[6]);
  //int workload_size = 100000;;
  int test_workload_size = atoi(argv[7]);
    
  // Read keys from file. Keys are in random order (not sorted).
  auto keys = new K[num_records];
  std::ifstream is(keys_file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is.is_open()) {
    std::cout << "Run `sh download.sh` to download the keys file" << std::endl;
    return 0;
  }
  is.read(reinterpret_cast<char*>(keys),
          std::streamsize(num_records * sizeof(K)));
  is.close();

  // Combine loaded keys with randomly generated values
  std::vector<std::pair<K, V>> data(num_records);
  std::mt19937_64 gen_payload(std::random_device{}());
  for (int i = 0; i < num_records; i++) {
    data[i].first = keys[i];
    data[i].second = static_cast<V>(gen_payload());
  }
  delete[] keys;
  
  // Read weights
  std::vector<double> weights = read_weights(weights_file_path, num_records);

  // Read workloads
  std::vector<K> test_workload = read_workload(test_workload_file_path, test_workload_size);
    
  // Build index index
  /*
  std::cout << "Building learned index with " << num_second_level_models
            << " second level models..." << std::endl;
  */

  LookUpTableLearnedIndex<K, V> index(data, weights);
  auto build_start_time = std::chrono::high_resolution_clock::now();
  index.build(num_second_level_models, tableSize);
  double build_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - build_start_time)
          .count();

  // Run workload using learned index
  /*
  std::cout << "Running query workload..." << std::endl;
  auto workload_start_time = std::chrono::high_resolution_clock::now();
  V sum = 0;
  for (const auto& record : data) {
    K key = record.first;
    const V* payload = index.get_value(key);
    if (payload) {
      sum += *payload;
    }
  }
  double workload_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - workload_start_time)
          .count();
  */
  index.reset_last_mile_search_count();
  auto workload_start_time = std::chrono::high_resolution_clock::now();
  for (K key: test_workload) {
    const V* payload = index.get_value(key);
    if (!payload) {
      exit(1);
    }
  }
  double workload_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - workload_start_time)
          .count();
  int num_last_mile_search = index.get_last_mile_search_count();

  /*
  std::cout << "Workload complete. Learned index build time: "
            << build_time / 1e9
            << " seconds, workload time: " << workload_time / 1e9
            << " seconds, proof of work: " << sum << std::endl;
  */

  // output index build time and workload time on test workload
  int model_size = sizeof(index);  //bytes
  std::cout << model_size << "\t" << build_time / 1e9 << "\t" << workload_time / 1e9 << "\t" << num_last_mile_search << std::endl;
}
