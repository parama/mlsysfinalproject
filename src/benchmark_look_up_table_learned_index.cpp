#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#include "look_up_table_learned_index.h"

#define K double
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

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Incorrect usage." << std::endl;
    exit(1);
  }

  int num_second_level_models = atoi(argv[1]);
  int tableSize = atoi(argv[2]);
  std::string keys_file_path = std::string(argv[3]);
  std::string workload_file_path = std::string(argv[4]);
  std::string test_workload_file_path = std::string(argv[5]);
  //int num_records = 200000000;
  int num_records = atoi(argv[6]);
  //int workload_size = 100000;;
  int workload_size = atoi(argv[7]);
  int test_workload_size = atoi(argv[8]);
    
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
    
  // Read workloads
  std::vector<K> workload = read_workload(workload_file_path, workload_size);
  std::vector<K> test_workload = read_workload(test_workload_file_path, test_workload_size);
    
  // Build index index
  /*
  std::cout << "Building learned index with " << num_second_level_models
            << " second level models..." << std::endl;
  */
  LookUpTableLearnedIndex<K, V> index(data, workload);
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

  /*
  std::cout << "Workload complete. Learned index build time: "
            << build_time / 1e9
            << " seconds, workload time: " << workload_time / 1e9
            << " seconds, proof of work: " << sum << std::endl;
  */

  // output index build time and workload time on test workload
  int model_size = sizeof(index);  //bytes
  std::cout << model_size << "\t" << build_time / 1e9 << "\t" << workload_time / 1e9 << std::endl;
}
