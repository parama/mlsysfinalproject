#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#include "weighted_learned_index.h"

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

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << "Incorrect usage." << std::endl;
    exit(1);
  }

  int num_second_level_models = atoi(argv[1]);
  std::string keys_file_path = std::string(argv[2]);
  std::string weights_file_path = std::string(argv[3]);
  std::string test_workload_file_path = std::string(argv[4]);
  //int num_records = 200000000;
  int num_records = atoi(argv[5]);
  //int workload_size = 100000;;
  int test_workload_size = atoi(argv[6]);
        
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

  // Read weights generated from workload
  auto weights = new K[num_records];
  std::ifstream is_weight(weights_file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is_weight.is_open()) {
    std::cout << "Run `python generate_workflow` then `convert_workload_to_weights` to generate weights" << std::endl;
    return 0;
  }
        
  is_weight.read(reinterpret_cast<char*>(weights),
          std::streamsize(num_records * sizeof(K)));
  is_weight.close();
    
  // Combine loaded keys with randomly generated values
  std::vector<std::tuple<K, V, K>> data(num_records);
  std::mt19937_64 gen_payload(std::random_device{}());
  for (int i = 0; i < num_records; i++) {
    std::get<0>(data[i]) = keys[i];
    std::get<1>(data[i]) = static_cast<V>(gen_payload());
    std::get<2>(data[i]) = weights[i];
  }
  delete[] keys;
  delete[] weights;

  // read test workload
  std::vector<K> test_workload = read_workload(test_workload_file_path, test_workload_size);

  // Build index index
  /*
  std::cout << "Building learned index with " << num_second_level_models
            << " second level models..." << std::endl;
  */
  WLearnedIndex<K, V> index(data);
  auto build_start_time = std::chrono::high_resolution_clock::now();
  index.build(num_second_level_models);
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
    K key = std::get<0>(record);
    const V* payload = index.get_value(key);
    if (payload) {
      sum += *payload;
    }
  }
  double workload_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - workload_start_time)
          .count();

  std::cout << "Workload complete. Learned index build time: "
            << build_time / 1e9
            << " seconds, workload time: " << workload_time / 1e9
            << " seconds, proof of work: " << sum << std::endl;
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
  
  int model_size = sizeof(index);  //bytes
  std::cout << model_size << "\t" << build_time / 1e9 << "\t" << workload_time / 1e9 << "\t" << num_last_mile_search << std::endl;
}
