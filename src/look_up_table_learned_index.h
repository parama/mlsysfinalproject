#pragma once

#include <iostream>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include "linear_model.h"

template <class K, class V>
class LookUpTableLearnedIndex {
  static_assert(std::is_arithmetic<K>::value,
                "Learned index key type must be numeric.");

 public:
  typedef std::pair<K, V> record;

  LookUpTableLearnedIndex(std::vector<record> data, std::vector<K> workload) : data_(data), workload_(workload) {
    std::sort(data_.begin(), data_.end());
  }

  // borrowed from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  std::vector<int> sort_indexes(std::vector<int> &v) {

    // initialize original index locations
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    std::stable_sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) {return v[i1] < v[i2];});

    return idx;
  }

  // Build a two-level RMI that only uses linear regression models, with the
  // specified number of second-level models.
  void build(int num_second_level_models, int tableSize) {
    assert(num_second_level_models > 0);
    second_level_models_.clear();

    // Construct the root model over the entire data.
    // Extract keys from key-value records. In practice, you would want to avoid
    // this because it requires making a redundant temporary copy of all the
    // keys, but for simplicity in this lab we will make some redundant copies.
    std::vector<K> keys;
    std::transform(std::begin(data_), std::end(data_), std::back_inserter(keys),
                   [](auto const& pair) { return pair.first; });
      
    // Build a vector of positions (i.e., indexes) for each key.
    std::vector<int> positions(keys.size());
    std::iota(std::begin(positions), std::end(positions), 0);

    // compute weights for each key in the set
    std::unordered_map<K, int> key_to_position;
    int key_size = keys.size();
    int workload_size = workload_.size();
    for (int i = 0; i < key_size; i++) {
      if (key_to_position.find(keys[i]) == key_to_position.end()) {
        key_to_position[keys[i]] = i;
      }
    }
    std::vector<int> weights;
    for (int i = 0; i < key_size; i++) {
      weights.push_back(0);
    }
    for (int i = 0; i < workload_size; i ++) {
      weights[key_to_position[workload_[i]]] = weights[key_to_position[workload_[i]]] + 1;      
    }

    // Create a look-up table for the top most frequent keys
    std::vector<int> sorted_idx = sort_indexes(weights);
    int idx_size = sorted_idx.size();
    std::unordered_set<int> idx_to_save;
    if (tableSize >= idx_size) {
      tableSize = idx_size;
    }
    for (int i = 0; i < tableSize; i++) {
      idx_to_save.insert(sorted_idx[idx_size - 1 - i]);
    }

    std::vector<K> keys_to_train;
    std::vector<int> positions_to_train;
    look_up_table_.clear();
    int all_keys_size = keys.size();
    for (int i = 0; i < all_keys_size; i ++) {
      if (idx_to_save.find(i) != idx_to_save.end()) {
        // add to look-up table
        if (look_up_table_.find(keys[i]) == look_up_table_.end()){
          look_up_table_[keys[i]] = positions[i];
        }
      } else {
        keys_to_train.push_back(data_[i].first);
        positions_to_train.push_back(data_[i].second);
      }
    }

    
    // For the root node over n records, these are simply the integers 0 through
    // n-1.
    root_model_.train(keys_to_train, positions_to_train);
    // Rescale the root model so that instead of predicting a position, it
    // predicts the index for the second-level model. Feeding a key through
    // the root model will output the index of the second-level model to which
    // the key should be assigned (root model outputs may need to be manually
    // bounded between 0 and num_second_level_models-1).
    root_model_.rescale(static_cast<double>(num_second_level_models) /
                        keys_to_train.size());

    // Use the trained root model to assign records to each of the second-level
    // models. Then train the second-level models to predict the positions for
    // each of their assigned records and compute the maximum prediction error
    // for each second-level model. Unlike the paper, in which each model
    // stores both a min-error (i.e., a left-error) and a max-error (i.e., a
    // right error), here we will only store a single maximum bi-directional
    // error for each second-level model.
      
    std::vector<K> bucket_keys;
    std::vector<int> bucket_positions;
      
    int start_pos;
    int end_pos = 0;  // exclusive
    for (int i = 0; i < num_second_level_models; i++) {
      start_pos = end_pos;
      while (end_pos < static_cast<int>(keys_to_train.size()) &&
             root_model_.predict(keys_to_train[end_pos]) <= i) {
        end_pos++;
      }
      // Edge case
      if (i == num_second_level_models - 1) {
        end_pos = static_cast<int>(keys_to_train.size());
      }
      bucket_keys.clear();
      bucket_positions.clear();
      for (int i = start_pos; i < end_pos; i++) {
        bucket_keys.push_back(keys_to_train[i]);
        bucket_positions.push_back(positions_to_train[i]);
      }
      LinearModel<K> model;
      model.train(bucket_keys, bucket_positions);
      second_level_models_.push_back(model);

      // Compute error bound
      int max_error = 0;
      for (int pos = start_pos; pos < end_pos; pos++) {
        int predicted_pos = model.predict(bucket_keys[pos]);
        max_error = std::max(max_error, std::abs(pos - predicted_pos));
      }
      second_level_error_bounds_.push_back(max_error);
    }
  }

  // If the key exists, return a pointer to the corresponding value in data_.
  // If the key does not exist, return a nullptr.
  V* get_value(K key) {
    assert(second_level_models_.size() > 0);

    // check if the key is inside the look-up table
    if (look_up_table_.find(key) != look_up_table_.end()) {
      return &data_[look_up_table_.at(key)].second;
    }

    int root_model_output = root_model_.predict(key);

    // Use the root model's output to select a second-level model, then use
    // the second-level model to predict the key's position, then do a
    // last-mile search using the model's error bound to find the true position
    // of the key. If the key exists, return a pointer to the value. If the
    // key does not exist, return a nullptr.
    // NOTE: to receive full credit, the last-mile search should use the
    // `last_mile_search` method provided below.
      
    int second_level_model_index =
        std::max(0, std::min(static_cast<int>(second_level_models_.size()) - 1,
                             root_model_output));
      
    const auto& second_level_model =
          second_level_models_[second_level_model_index];
        
    int predicted_position = second_level_model.predict(key);
        
    if (data_[predicted_position].first == key) {
      return &data_[predicted_position].second;
    } else {
      last_mile_search_count_ = last_mile_search_count_ + 1;
    }
        
    int error_bound = second_level_error_bounds_[second_level_model_index];
    int bound_start_pos =
        std::max(0, std::min(static_cast<int>(data_.size()),
                             predicted_position - error_bound));
    int bound_end_pos =
        std::max(0, std::min(static_cast<int>(data_.size()),
                             predicted_position + error_bound + 1));
    int true_position = last_mile_search(key, bound_start_pos, bound_end_pos);
    if (true_position == -1) {
      return nullptr;
    } else {
      return &(data_[true_position].second);
    }
  }

  int get_last_mile_search_count() {
    return last_mile_search_count_;
  }

  void reset_last_mile_search_count() {
    last_mile_search_count_ = 0;
  }

 private:
  // Do a binary search for the position of a key in the data.
  // Only search in the range between the given start position (inclusive)
  // and end position (exclusive).
  // If the key is not found in the data, return -1.
  int last_mile_search(K key, int start_pos, int end_pos) const {
    int pos = std::lower_bound(
                  data_.begin() + start_pos, data_.begin() + end_pos, key,
                  [](auto const& pair, K key) { return pair.first < key; }) -
              data_.begin();
    if (pos > static_cast<int>(data_.size()) || data_[pos].first != key) {
      return -1;
    } else {
      return pos;
    }
  }

  std::unordered_map<K, int>  look_up_table_;
  std::vector<record> data_;
  std::vector<K> workload_;
  LinearModel<K> root_model_;
  std::vector<LinearModel<K>> second_level_models_;
  // The maximum prediction error for each second-level model.
  std::vector<int> second_level_error_bounds_;
  int last_mile_search_count_;
};
