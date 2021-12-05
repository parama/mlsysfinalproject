#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

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
  if (argc != 5) {
    std::cout << "Incorrect usage." << std::endl;
    exit(1);
  }

  std::string keys_file_path = std::string(argv[1]);
  std::string test_workload_file_path = std::string(argv[2]);
  //int num_records = 200000000;
  int num_records = atoi(argv[3]);
  //int workload_size = 100000;;
  int test_workload_size = atoi(argv[4]);

  // Read keys from file. Keys are in random order (not sorted).
  std::vector<K> keys(num_records);
  std::ifstream is(keys_file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is.is_open()) {
    std::cout << "Run `sh download.sh` to download the keys file" << std::endl;
    return 0;
  }
  is.read(reinterpret_cast<char*>(keys.data()),
          std::streamsize(num_records * sizeof(K)));
  is.close();

  // Combine loaded keys with randomly generated values
  std::vector<std::pair<K, V>> data(num_records);
  std::mt19937_64 gen_payload(std::random_device{}());
  for (int i = 0; i < num_records; i++) {
    data[i].first = keys[i];
    data[i].second = static_cast<V>(gen_payload());
  }

  // read test workload
  std::vector<K> test_workload = read_workload(test_workload_file_path, test_workload_size);

  // Sort data
  // std::cout << "Sorting data..." << std::endl;
  std::sort(data.begin(), data.end());

  // Run workload using binary search
  /* std::cout << "Running query workload..." << std::endl;
  auto workload_start_time = std::chrono::high_resolution_clock::now();
  V sum = 0;
  for (K key : keys) {
    auto it = std::lower_bound(
        data.begin(), data.end(), key,
        [](auto const& pair, K key) { return pair.first < key; });
    sum += it->second;
  }
  double workload_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - workload_start_time)
          .count();

  std::cout << "Workload complete. Binary search workload time: "
            << workload_time / 1e9 << " seconds, proof of work: " << sum
            << std::endl;
  */
  auto workload_start_time = std::chrono::high_resolution_clock::now();
  for (K key: test_workload) {
    std::lower_bound(
        data.begin(), data.end(), key,
        [](auto const& pair, K key) { return pair.first < key; });
  }
  double workload_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - workload_start_time)
          .count();
  
  std::cout << "NA" << "\t" << "NA" << "\t" << workload_time / 1e9 << "\t" << "NA" << std::endl;
}
