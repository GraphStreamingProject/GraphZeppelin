#pragma once
#include <iostream>

// Configuration options for the minimum cut sketch algorithm
class MCAlgConfiguration {
 private:
  // How large to make update batches as factor of sketch size
  double _batch_factor = 1;
  
  // Returned min-cut guaranteed to be a +/- epsilon multiplicative approx of the true min cut.
  double _epsilon = 0.5;

  // Number of subgraphs for which we use a delta sketch
  // When applying sketch updates to other subgraphs, apply updates directly to sketch
  size_t _num_subgraphs_use_delta = 2;

  friend class MinCutSketchAlg;
 public:
  // setters
  MCAlgConfiguration& batch_factor(double batch_factor) {
    if (batch_factor <= 0) {
      std::cerr << "WARNING: Batch factor in MCAlgConfiguration must be > 0." << std::endl;
      std::cerr << "         Setting to default value: " << _batch_factor << std::endl;
    } else {
      _batch_factor = batch_factor;
    }
    return *this;
  }
  MCAlgConfiguration& epsilon(double epsilon) {
    if (epsilon <= 0 || epsilon > 1) {
      std::cerr << "WARNING: MCAlgConfiguration epsilon must be in range (0, 1]." << std::endl;
      std::cerr << "         Setting to default value: " << _epsilon << std::endl;
    } else {
      _epsilon = epsilon;
    }
    return *this;
  }
  MCAlgConfiguration& num_subgraphs_use_delta(size_t num_subgraphs) {
    _num_subgraphs_use_delta = num_subgraphs;
    return *this;
  }

  // getters
  double get_batch_factor() { return _batch_factor; }
  double get_epsilon() { return _epsilon; }
  size_t get_num_subgraphs_use_delta() { return _num_subgraphs_use_delta; }

  friend std::ostream& operator<< (std::ostream &out, const MCAlgConfiguration &conf) {
    out << "Minimum Cut Algorithm Configuration:" << std::endl;
    out << "  batch_factor = " << conf._batch_factor << std::endl;
    return out;
  }
};
