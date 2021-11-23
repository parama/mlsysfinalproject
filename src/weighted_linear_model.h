#pragma once

#include <vector>
#include <numeric>
#include <cmath>

/* A simple linear regression model for predicting the location of a given key
 * in a sorted array: y = m * x + b, where m is the slope and b is the
 * intercept. The key x can be any numeric type (e.g., integer or floating
 * point). The predicted position y is an integer.
 */
template <class K, class V>

class WLinearModel {
  static_assert(std::is_arithmetic<K>::value,
                "Linear model feature type must be numeric.");

 public:
  WLinearModel() = default;
  double m_ = 0;  // slope
  double b_ = 0;  // intercept

  // Train the model (i.e., set values for the slope m_ and the intercept b_)
  // to predict the positions from the keys. In other words, `keys` is a vector
  // of scalar model inputs, and `positions` is the vector of the corresponding
  // desired model outputs (e.g., for input `keys[0]`, the desired output is
  // `positions[0]`).
  void train(const std::vector<K>& keys, const std::vector<int>& positions,
             const std::vector<V>& workloads) {
    assert(keys.size() == positions.size());
    assert(keys.size() == workloads.size());
      
    double n = keys.size(); // number elements
    int tot_workload = std::accumulate(workloads.begin(), workloads.end(), 0);
    
    // keys to search for in the index (x)
    // positions to retrieve (y)
      
    double x_mean = 0;
    double y_mean = 0;
      
    for (int i = 0; i < n; i++) {
        x_mean += workloads[i] * keys[i];
        y_mean += workloads[i] * positions[i];
    }
      
    x_mean = x_mean / tot_workload;
    y_mean = y_mean / tot_workload;
      
    if (n <= 1) {
        m_ = 0;
        b_ = y_mean;
        return;
    }
      
    double numerator = 0;
    double denominator = 0;
      
    for (int i = 0; i < n; i++) {
        numerator += workloads[i] * (keys[i] - x_mean) * (positions[i] - y_mean);
        denominator += workloads[i] * pow(keys[i] - x_mean, 2);
    }

    if (denominator == 0) {
        m_ = 0;
        b_ = y_mean;
        return;
    }
    
    m_ = numerator / denominator;
    b_ = y_mean - m_ * x_mean;
      
  }

  int predict(K key) const {
    return static_cast<int>(m_ * static_cast<double>(key) + b_);
  }

  K inverse_predict(int position) const {
    return static_cast<K>((static_cast<double>(position) - b_) / m_);
  }

  void rescale(double scaling_factor) {
    m_ *= scaling_factor;
    b_ *= scaling_factor;
  }

 private:
  double x_ = 0;  // slope
  double y_ = 0;  // intercept
};
