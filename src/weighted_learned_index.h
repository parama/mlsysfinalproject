#pragma once

#include <iostream>
#include <numeric>
#include <vector>

#include "weighted_linear_model.h"

template <class K, class V>
class WLearnedIndex {
  static_assert(std::is_arithmetic<K>::value,
                "Learned index key type must be numeric.");

 public:
  typedef std::tuple<K, V, K> record;

  WLearnedIndex(std::vector<record> data) : data_(data) {
    std::sort(data_.begin(), data_.end());
  }

  // Build a two-level RMI that only uses linear regression models, with the
  // specified number of second-level models.
  void build(int num_second_level_models) {
    assert(num_second_level_models > 0);
    second_level_models_.clear();

    // Construct the root model over the entire data.
    // Extract keys from key-value records. In practice, you would want to avoid
    // this because it requires making a redundant temporary copy of all the
    // keys, but for simplicity in this lab we will make some redundant copies.
    std::vector<K> keys;
    std::transform(std::begin(data_), std::end(data_), std::back_inserter(keys),
                   [](auto const& tuple) { return std::get<0>(tuple); });
      
    std::vector<V> workload;
    std::transform(std::begin(data_), std::end(data_), std::back_inserter(workload),
                   [](auto const& tuple) { return std::get<2>(tuple); });
      
    // Build a vector of positions (i.e., indexes) for each key.
    // For the root node over n records, these are simply the integers 0 through
    // n-1.
    std::vector<int> positions(keys.size());
    std::iota(std::begin(positions), std::end(positions), 0);
    root_model_.train(keys, positions, workload);
    // Rescale the root model so that instead of predicting a position, it
    // predicts the index for the second-level model. Feeding a key through
    // the root model will output the index of the second-level model to which
    // the key should be assigned (root model outputs may need to be manually
    // bounded between 0 and num_second_level_models-1).
    root_model_.rescale(static_cast<double>(num_second_level_models) /
                        keys.size());

    // Use the trained root model to assign records to each of the second-level
    // models. Then train the second-level models to predict the positions for
    // each of their assigned records and compute the maximum prediction error
    // for each second-level model. Unlike the paper, in which each model
    // stores both a min-error (i.e., a left-error) and a max-error (i.e., a
    // right error), here we will only store a single maximum bi-directional
    // error for each second-level model.
      
    int keys_size = keys.size();
    for (int i = 0; i < num_second_level_models; i++) {
        WLinearModel<K, V> model;
        second_level_models_.push_back(model);
        
        int error = keys_size;
        second_level_error_bounds_.push_back(error);
    }
      
    for (int i = 0; i < num_second_level_models; i++) {
        int bucket_index = i;
        std::vector<K> bucket_keys;
        std::vector<int> bucket_positions;
        std::vector<V> bucket_workloads;
        
        // get the data that belongs to the bucket
        for (int j = 0; j < keys_size; j++) {
            int predicted_bucket = root_model_.predict(keys[j]);
            // clip
            predicted_bucket = std::max<int>(predicted_bucket, 0);
            predicted_bucket = std::min<int>(predicted_bucket, num_second_level_models - 1);
            if (predicted_bucket == bucket_index) {
                bucket_keys.push_back(keys[j]);
                bucket_positions.push_back(positions[j]);
                bucket_workloads.push_back(workload[j]);
            }
        }
       
        // train the second-level model on subset of data
        int bucket_size = bucket_keys.size();
        second_level_models_[bucket_index].train(bucket_keys, bucket_positions, bucket_workloads);
        
        int max_error = 0;
        for (int j = 0; j < bucket_size; j++) {
            int predicted_index = second_level_models_[bucket_index].predict(bucket_keys[j]);
            // clip
            predicted_index = std::max<int>(predicted_index, 0);
            predicted_index = std::min<int>(predicted_index, keys_size - 1);
            int error = std::abs(predicted_index - bucket_positions[j]);
            if (error > max_error) {
                max_error = error;
            }
        }
        second_level_error_bounds_[bucket_index] = max_error;
    }
  }

  // If the key exists, return a pointer to the corresponding value in data_.
  // If the key does not exist, return a nullptr.
  const V* get_value(K key) const {
    assert(second_level_models_.size() > 0);
    int root_model_output = root_model_.predict(key);

    // Use the root model's output to select a second-level model, then use
    // the second-level model to predict the key's position, then do a
    // last-mile search using the model's error bound to find the true position
    // of the key. If the key exists, return a pointer to the value. If the
    // key does not exist, return a nullptr.
    // NOTE: to receive full credit, the last-mile search should use the
    // `last_mile_search` method provided below.
      
    int num_second_level_models = second_level_models_.size();
    int second_level_index = std::max<int>(root_model_output, 0);
    second_level_index = std::min<int>(second_level_index, num_second_level_models - 1);
    
    int data_size = data_.size();
    int predicted_index = second_level_models_[second_level_index].predict(key);
    predicted_index = std::max<int>(predicted_index, 0);
    predicted_index = std::min<int>(predicted_index, data_size - 1);
      
    int error_bound = second_level_error_bounds_[second_level_index];
    int start_search = predicted_index - error_bound;
    int end_search = predicted_index + error_bound;
    //clip
    start_search = std::max<int>(start_search, 0);
    end_search = std::max<int>(end_search, 0);
    start_search = std::min<int>(start_search, data_size);
    end_search = std::min<int>(end_search, data_size);
      
    int pos = last_mile_search(key, start_search, end_search);
    if (pos == -1) {
        return nullptr;
    }
    return &std::get<1>(data_[pos]);
  }

 private:
  // Do a binary search for the position of a key in the data.
  // Only search in the range between the given start position (inclusive)
  // and end position (exclusive).
  // If the key is not found in the data, return -1.
  int last_mile_search(K key, int start_pos, int end_pos) const {
    int pos = std::lower_bound(
                  data_.begin() + start_pos, data_.begin() + end_pos, key,
                  [](auto const& tuple, K key) { return std::get<0>(tuple) < key; }) -
              data_.begin();
    if (pos > static_cast<int>(data_.size()) || std::get<0>(data_[pos]) != key) {
      return -1;
    } else {
      return pos;
    }
  }

  std::vector<record> data_;
  WLinearModel<K, V> root_model_;
  std::vector<WLinearModel<K, V>> second_level_models_;
  // The maximum prediction error for each second-level model.
  std::vector<int> second_level_error_bounds_;
};
