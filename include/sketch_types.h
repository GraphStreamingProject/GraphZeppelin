#pragma once

#include <exception>
#include <vector>

#include "types.h"
// enum SerialType {
//   FULL,
//   RANGE,
//   SPARSE,
// };

enum SampleResult {
  GOOD,  // sampling this sketch returned a single non-zero value
  ZERO,  // sampling this sketch returned that there are no non-zero values
  FAIL   // sampling this sketch failed to produce a single non-zero value
};

struct SketchSample {
  vec_t idx;
  SampleResult result;
};

struct ExhaustiveSketchSample {
  std::vector<vec_t> idxs;
  SampleResult result;
};

class OutOfSamplesException : public std::exception {
 private:
  std::string err_msg;

 public:
  OutOfSamplesException(size_t seed, size_t num_samples, size_t sample_idx)
      : err_msg("This sketch (seed=" + std::to_string(seed) +
                ", max samples=" + std::to_string(num_samples) +
                ") cannot be sampled more times (cur idx=" + std::to_string(sample_idx) + ")!") {}
  virtual const char* what() const throw() { return err_msg.c_str(); }
};
