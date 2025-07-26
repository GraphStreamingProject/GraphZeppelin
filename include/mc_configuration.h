#pragma once
#include <iostream>

// Configuration options for the minimum cut sketch algorithm
class MCAlgConfiguration {
 private:
  // How large to make update batches as factor of sketch size
  double _batch_factor = 1;
  
  // Returned min-cut guaranteed to be a +/- epsilon multiplicative approx of the true min cut.
  double _epsilon = 0.5;

  // Number of sketch subgraphs to create immediately.
  // Preallocating these sketches improves performance at the cost of additional memory consumption
  // if you already that you will need more (say 5) then setting this to the higher value is good
  size_t _initial_subgraphs = 1;

  friend class MinCutSketchAlg;
 public:
  // setters
  MCAlgConfiguration& batch_factor(double batch_factor) {
    if (batch_factor <= 0) {
      std::cerr << "WARNING: MCAlgConfiguration, batch factor must be > 0." << std::endl;
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
  MCAlgConfiguration& initial_subgraphs(size_t num_subgraphs) {
    if (num_subgraphs == 0) {
      std::cerr << "WARNING: MCAlgConfiguration, initial subgraphs must be > 0." << std::endl;
      std::cerr << "         Setting to default value: " << _initial_subgraphs << std::endl;
    } else {
      _initial_subgraphs = num_subgraphs;
    }
    return *this;
  }

  // getters
  double get_batch_factor() { return _batch_factor; }
  double get_epsilon() { return _epsilon; }
  size_t get_initial_subgraphs() { return _initial_subgraphs; }

  friend std::ostream& operator<< (std::ostream &out, const MCAlgConfiguration &conf) {
    out << "Minimum Cut Algorithm Configuration:" << std::endl;
    out << "  batch_factor             = " << conf._batch_factor << std::endl;
    out << "  epsilon                  = " << conf._epsilon << std::endl;
    out << "  initial_sketch_subgraphs = " << conf._initial_subgraphs << std::endl;
    return out;
  }
};
