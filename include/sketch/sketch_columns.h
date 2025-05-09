#pragma once
#include "bucket.h"
#include "sketch_concept.h"

#include "util.h"

#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>

/*
 * FOR NOW - simplest possible design
*/
class FixedSizeSketchColumn {
private:

  std::unique_ptr<Bucket[]> buckets;
  Bucket deterministic_bucket = {0, 0};
  uint16_t col_idx; // determines column seeding
  uint8_t capacity;
public:
  static uint64_t seed;
  static void set_seed(uint64_t new_seed) {
    seed = new_seed;
  };  
  static const uint64_t get_seed() {
    return seed;
  };

  FixedSizeSketchColumn(uint8_t capacity, uint16_t col_idx);
  FixedSizeSketchColumn(const FixedSizeSketchColumn &other);
  ~FixedSizeSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void merge(FixedSizeSketchColumn &other);
  uint8_t get_depth() const;
  void serialize(std::ostream &binary_out) const;
  void reset_sample_state() {
    //no-op
  };
  bool operator==(const FixedSizeSketchColumn &other) const {
    for (size_t i = 0; i < capacity; ++i) {
      if (buckets[i] != other.buckets[i]) {
        return false;
      }
    }
    return true;
  }
  friend std::ostream& operator<<(std::ostream &os, const FixedSizeSketchColumn &sketch) {
    os << "FixedSizeSketchColumn: " << std::endl;
    os << "Capacity: " << (int)sketch.capacity << std::endl;
    os << "Column Index: " << (int)sketch.col_idx << std::endl;
    os << "Deterministic Bucket: " << sketch.deterministic_bucket << std::endl;
    for (size_t i = 0; i < sketch.capacity; ++i) {
      os << "Bucket[" << i << "]: " << sketch.buckets[i] << std::endl;
    }
    return os;
  }
};


class ResizeableSketchColumn {
private:
  static uint64_t seed;
  hwy::AlignedFreeUniquePtr<Bucket[]> aligned_buckets;
  Bucket deterministic_bucket = {0, 0};
  uint16_t col_idx; // determines column seeding
  uint8_t capacity;
public:
  static void set_seed(uint64_t new_seed) { seed = new_seed; };
  static const uint64_t get_seed() { return seed; };

  ResizeableSketchColumn(uint8_t start_capacity, uint16_t col_idx);
  ResizeableSketchColumn(const ResizeableSketchColumn &other);
  ~ResizeableSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void merge(ResizeableSketchColumn &other);
  uint8_t get_depth() const;
  void reset_sample_state() {
    //no-op
  };
  void serialize(std::ostream &binary_out) const;
  bool operator==(const ResizeableSketchColumn &other) const {
    for (size_t i = 0; i < capacity; ++i) {
      if (aligned_buckets[i] != other.aligned_buckets[i]) {
        return false;
      }
    }
    return true;
  }
private:
  void reallocate(uint8_t new_capacity);
};

