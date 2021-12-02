#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#include "weighted_learned_index.h"

#define K double
#define V int64_t

int main(int argc, char** argv) {
  int num_second_level_models = 100;
  if (argc > 1) {
    num_second_level_models = atoi(argv[1]);
  }

  std::string keys_file_path = "SOSD/data/wiki_ts_200M_uint64";
  std::string weights_file_path = "data/weights/wiki_ts_200M_uint64_workload100k_alpha1.1";
  int num_records = 200000000;
        
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

  // Build index index
  std::cout << "Building learned index with " << num_second_level_models
            << " second level models..." << std::endl;
  WLearnedIndex<K, V> index(data);
  auto build_start_time = std::chrono::high_resolution_clock::now();
  index.build(num_second_level_models);
  double build_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - build_start_time)
          .count();

  // Run workload using learned index
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
}
