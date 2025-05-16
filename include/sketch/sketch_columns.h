#pragma once
#include "bucket.h"
#include "sketch_concept.h"

#include <cmath>

#include "util.h"

#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>

/*
 * FOR NOW - simplest possible design
*/
class FixedSizeSketchColumn {
private:
  Bucket *buckets;
  Bucket deterministic_bucket = {0, 0};
  uint64_t seed;
  uint8_t capacity;
public:
  void set_seed(uint64_t new_seed) {
    seed = new_seed;
  };  
  uint64_t get_seed() const {
    return seed;
  };

  FixedSizeSketchColumn(uint8_t capacity, uint64_t seed);
  FixedSizeSketchColumn(const FixedSizeSketchColumn &other);
  FixedSizeSketchColumn(FixedSizeSketchColumn &&other);
  ~FixedSizeSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  
  void update(const vec_t update);
  void merge(FixedSizeSketchColumn &other);
  uint8_t get_depth() const;
  void serialize(std::ostream &binary_out) const;
  
  static uint8_t suggest_capacity(size_t num_indices) { 
    return static_cast<uint8_t>(2 + ceil(log2(num_indices)));
  }

  void reset_sample_state() {
    //no-op
  };

  [[deprecated]]
  void zero_contents() {
    clear();
  }

  bool operator==(const FixedSizeSketchColumn &other) const {
    for (size_t i = 0; i < capacity; ++i) {
      if (buckets[i] != other.buckets[i]) {
        return false;
      }
    }
    return true;
  }
  // move assignment operator
  FixedSizeSketchColumn& operator=(FixedSizeSketchColumn &&other);

  friend std::ostream& operator<<(std::ostream &os, const FixedSizeSketchColumn &sketch) {
    os << "FixedSizeSketchColumn: " << std::endl;
    os << "Capacity: " << (int)sketch.capacity << std::endl;
    os << "Column Seed: " << (int)sketch.seed << std::endl;
    os << "Deterministic Bucket: " << sketch.deterministic_bucket << std::endl;
    for (size_t i = 0; i < sketch.capacity; ++i) {
      os << "Bucket[" << i << "]: " << sketch.buckets[i] << std::endl;
    }
    return os;
  }

};


class ResizeableSketchColumn {
private:
  Bucket *buckets;
  Bucket deterministic_bucket = {0, 0};
  uint64_t seed;
  uint8_t capacity;
public:
  void set_seed(uint64_t new_seed) { seed = new_seed; };
  uint64_t get_seed() const { return seed; };

  ResizeableSketchColumn(uint8_t start_capacity, uint64_t seed);
  ResizeableSketchColumn(const ResizeableSketchColumn &other);
  ResizeableSketchColumn(ResizeableSketchColumn &&other);
  ResizeableSketchColumn& operator=(ResizeableSketchColumn &&other);
  ~ResizeableSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void merge(ResizeableSketchColumn &other);
  uint8_t get_depth() const;

  [[deprecated]]
  void zero_contents() {
    clear();
  }

  void reset_sample_state() {
    //no-op
  };

  static uint8_t suggest_capacity(size_t num_indices) {
    return 4;
  }
  
  void serialize(std::ostream &binary_out) const;
  
  friend std::ostream& operator<<(std::ostream &os, const ResizeableSketchColumn&sketch) {
    os << "ResizeableSketchColumn: " << std::endl;
    os << "Capacity: " << (int)sketch.capacity << std::endl;
    os << "Column Seed: " << (int)sketch.seed << std::endl;
    os << "Deterministic Bucket: " << sketch.deterministic_bucket << std::endl;
    for (size_t i = 0; i < sketch.capacity; ++i) {
      os << "Bucket[" << i << "]: " << sketch.buckets[i] << std::endl;
    }
    return os;
  }
  
  bool operator==(const ResizeableSketchColumn &other) const {
    size_t other_depth = other.get_depth();
    if (get_depth() != other_depth) {
      return false;
    }
    for (size_t i = 0; i < other_depth; ++i) {
      if (buckets[i] != other.buckets[i]) {
        return false;
      }
    }
    return true;
  }
private:
  void reallocate(uint8_t new_capacity);
};


class ResizeableAlignedSketchColumn {
private:
  hwy::AlignedFreeUniquePtr<Bucket[]> aligned_buckets;
  Bucket deterministic_bucket = {0, 0};
  uint64_t seed;
  uint8_t capacity;
public:
  void set_seed(uint64_t new_seed) { seed = new_seed; };
  const uint64_t get_seed() { return seed; };

  ResizeableAlignedSketchColumn(uint8_t start_capacity, uint64_t seed);
  ResizeableAlignedSketchColumn(const ResizeableAlignedSketchColumn &other);
  ResizeableAlignedSketchColumn(ResizeableAlignedSketchColumn &&other);
  ResizeableAlignedSketchColumn& operator=(ResizeableAlignedSketchColumn &&other);
  ~ResizeableAlignedSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void merge(ResizeableAlignedSketchColumn &other);
  uint8_t get_depth() const;

  [[deprecated]]
  void zero_contents() {
    clear();
  }

  void reset_sample_state() {
    //no-op
  };

  static uint8_t suggest_capacity(size_t num_indices) {
    return 4;
  }
  
  void serialize(std::ostream &binary_out) const;
  
  friend std::ostream& operator<<(std::ostream &os, const ResizeableAlignedSketchColumn&sketch) {
    os << "ResizeableSketchColumn: " << std::endl;
    os << "Capacity: " << (int)sketch.capacity << std::endl;
    os << "Column Seed: " << (int)sketch.seed << std::endl;
    os << "Deterministic Bucket: " << sketch.deterministic_bucket << std::endl;
    for (size_t i = 0; i < sketch.capacity; ++i) {
      os << "Bucket[" << i << "]: " << sketch.aligned_buckets[i] << std::endl;
    }
    return os;
  }
  
  bool operator==(const ResizeableAlignedSketchColumn &other) const {
    size_t other_depth = other.get_depth();
    if (get_depth() != other_depth) {
      return false;
    }
    for (size_t i = 0; i < other_depth; ++i) {
      if (aligned_buckets[i] != other.aligned_buckets[i]) {
        return false;
      }
    }
    return true;
  }
private:
  void reallocate(uint8_t new_capacity);
};

