#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>

#include "weighted_learned_index.h"
#include "learned_index.h"
#include "look_up_table_learned_index.h"

#define K double
#define V int64_t

std::tuple<double, double, V> learned_index(std::vector<std::pair<K, V>> data, int num_second_level_models) {
    std::cout << "Building learned index with " << num_second_level_models
              << " second level models..." << std::endl;
    LearnedIndex<K, V> index(data);
    
    // Building time
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
    
    return std::make_tuple(workload_time, build_time, sum);
}

std::tuple<double, double, V> weighted_learned_index(std::vector<std::pair<K, V>> data, int num_second_level_models) {
    std::cout << "Building weighted learned index with " << num_second_level_models
              << " second level models..." << std::endl;
    WLearnedIndex<K, V> index(data);
    
    // Building time
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
    
    std::cout << "Workload complete. Weighted learned index build time: "
              << build_time / 1e9
              << " seconds, workload time: " << workload_time / 1e9
              << " seconds, proof of work: " << sum << std::endl;
    
    return std::make_tuple(workload_time, build_time, sum);
}

std::tuple<double, double, V> lookup_table_learned_index(std::vector<std::pair<K, V>> data, int num_second_level_models, int tableSize) {
    std::cout << "Building lookup table learned index with " << num_second_level_models
              << " second level models and " << tableSize
              << " table size ..." << std::endl;
    LookUpTableLearnedIndex<K, V> index(data);
    
    // Building time
    auto build_start_time = std::chrono::high_resolution_clock::now();
    index.build(num_second_level_models, tableSize);
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

    
    std::cout << "Workload complete. Lookup table learned index build time: "
              << build_time / 1e9
              << " seconds, workload time: " << workload_time / 1e9
              << " seconds, proof of work: " << sum << std::endl;
    
    return std::make_tuple(workload_time, build_time, sum);
}

std::tuple<double, double, V> binary_search(std::vector<std::pair<K, V>> data) {
    double build_time = 0;

    // Run workload using binary search
    std::cout << "Running query workload using Binary Search..." << std::endl;
    auto workload_start_time = std::chrono::high_resolution_clock::now();
    V sum = 0;
    for (const auto& record : data) {
      K key = record.first;
      auto it = std::lower_bound(
          data.begin(), data.end(), key,
          [](auto const& pair, K key) { return pair.first < key; });
      sum += it->second;
    }

    double workload_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - workload_start_time)
            .count();
    
    std::cout << "Workload complete. Binary Search: "
              << build_time / 1e9
              << " seconds, workload time: " << workload_time / 1e9
              << " seconds, proof of work: " << sum << std::endl;
    
    return std::make_tuple(workload_time, build_time, sum);
}

int benchmark_dataset(const std::string& keys_file_path, const std::string& workload_file_path, int num_second_level_models, int table_size) {
  const int num_records = 200000000;
  
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

  auto [bs_workload_time, bs_build_time, bs_sum] = binary_search(data);
  auto [l_workload_time, l_build_time, l_sum] = learned_index(data, num_second_level_models);
  auto [wl_workload_time, wl_build_time, wl_sum] = weighted_learned_index(data, num_second_level_models);
  auto [t_workload_time, t_build_time, t_sum] = lookup_table_learned_index(data, num_second_level_models, table_size);
    
  // Write results
  std::ofstream results_file;
  results_file.open("results/benchmark_result.csv");
  results_file << "model, workload time, build time, proof of work\n"
    << "binary search," << bs_workload_time / 1e9 << "," << bs_build_time / 1e9 << "," << bs_sum << "\n"
    << "linear model," << l_workload_time / 1e9 << "," << l_build_time / 1e9 << "," << l_sum << "\n"
    << "weighted linear model," << wl_workload_time / 1e9 << "," << wl_build_time / 1e9 << "," << wl_sum << "\n"
    << "lookup table linear model," << t_workload_time / 1e9 << "," << t_build_time / 1e9 << "," << t_sum << "\n";
  results_file.close();
    
  return 0;
}

int main(int argc, char** argv) {
    int table_size = 5;
    int num_second_level_models = 100;
    if (argc > 1) {
        num_second_level_models = atoi(argv[1]); // changeable
    }

    std::string keys_file_path = "data/wiki_ts_200M_uint64";
    std::string workload_file_path = "data/workloads/wiki_ts_200M_uint64_workload100k_alpha1.1";
    
    benchmark_dataset(keys_file_path, workload_file_path, num_second_level_models, table_size);
}

