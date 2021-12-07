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
    int start_pos;
    int end_pos = 0;  // exclusive
    for (int i = 0; i < num_second_level_models; i++) {
      start_pos = end_pos;
      while (end_pos < static_cast<int>(data_.size()) &&
             root_model_.predict(std::get<0>(data_[end_pos])) <= i) {
        end_pos++;
      }
      // Edge case
      if (i == num_second_level_models - 1) {
        end_pos = static_cast<int>(data_.size());
      }
      keys.clear();
      std::transform(std::begin(data_) + start_pos, std::begin(data_) + end_pos,
                     std::back_inserter(keys),
                     [](auto const& tuple) { return std::get<0>(tuple); });
      workload.clear();
      std::transform(std::begin(data_) + start_pos, std::begin(data_) + end_pos,
                     std::back_inserter(workload),
                     [](auto const& tuple) { return std::get<2>(tuple); });
      positions.clear();
      positions.resize(end_pos - start_pos);
      std::iota(std::begin(positions), std::end(positions), start_pos);
      WLinearModel<K, V> model;
      model.train(keys, positions, workload);
      second_level_models_.push_back(model);

      // Compute error bound
      int max_error = 0;
      for (int pos = start_pos; pos < end_pos; pos++) {
        int predicted_pos = model.predict(std::get<0>(data_[pos]));
        max_error = std::max(max_error, std::abs(pos - predicted_pos));
      }
      second_level_error_bounds_.push_back(max_error);
      }
  }

  // If the key exists, return a pointer to the corresponding value in data_.
  // If the key does not exist, return a nullptr.
  V* get_value(K key) {
    assert(second_level_models_.size() > 0);
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
        
    if (std::get<0>(data_[predicted_position]) == key) {
      return &std::get<1>(data_[predicted_position]);
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
      return &std::get<1>(data_[true_position]);
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
  int last_mile_search_count_;
};
