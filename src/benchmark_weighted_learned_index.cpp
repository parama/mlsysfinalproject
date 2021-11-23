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

  std::string keys_file_path = "wiki_ts_200M_uint64";
  std::string workload_file_path = "wiki_ts_200M_uint64_workload";
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

  // Read workload
  auto workload = new V[num_records];
  std::ifstream is_workload(workload_file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is_workload.is_open()) {
    std::cout << "Run `python generate_workflow` to generate workloads" << std::endl;
    return 0;
  }
    
  is_workload.read(reinterpret_cast<char*>(workload),
          std::streamsize(num_records * sizeof(V)));
  is_workload.close();

  // Combine loaded keys with randomly generated values
  std::vector<std::pair<K, V>> data(num_records);
  for (int i = 0; i < num_records; i++) {
    data[i].first = keys[i];
    data[i].second = static_cast<V>(workload[i]);
  }
  delete[] keys;
  delete[] workload;

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

  std::cout << "Workload complete. Learned index build time: "
            << build_time / 1e9
            << " seconds, workload time: " << workload_time / 1e9
            << " seconds, proof of work: " << sum << std::endl;
}
