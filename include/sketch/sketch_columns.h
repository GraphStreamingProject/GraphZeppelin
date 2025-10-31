#pragma once
#include "bucket.h"
#include "sketch_concept.h"

#include <cmath>

#include "util.h"

#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>

#include <gtest/gtest.h>

// #include /* <folly/synchronization/RWSpinLock.h> */
#include "RWSpinLock.h"

/*
 * FOR NOW - simplest possible design
*/
class FixedSizeSketchColumn {
private:
  Bucket deterministic_bucket = {0, 0};
  Bucket *buckets;
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
  FixedSizeSketchColumn& operator=(const FixedSizeSketchColumn &other);

  FixedSizeSketchColumn(FixedSizeSketchColumn &&other) noexcept;
  FixedSizeSketchColumn& operator=(FixedSizeSketchColumn &&other) noexcept;

  ~FixedSizeSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  
  void prefetch();
  
  void update(const vec_t update);
  void atomic_update(const vec_t update);
  void merge(FixedSizeSketchColumn const& other);
  uint8_t get_depth() const;
  size_t space_usage_bytes() const;
  void serialize(std::ostream &binary_out) const;
  
  static uint8_t suggest_capacity(size_t num_indices) { 
    return static_cast<uint8_t>(2 + ceil(log2(num_indices)));
  }

  void reset_sample_state() {
    //no-op
  };
  
  const ColumnEntryDelta generate_entry_delta(vec_t update) const;
  void apply_entry_delta(const ColumnEntryDelta &delta);
  void atomic_apply_entry_delta(const ColumnEntryDelta &delta);
  

  inline bool is_initialized() const {
    return buckets != nullptr;
  }

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

FRIEND_TEST(SketchColumnTestSuite, TestMergeResizing);
FRIEND_TEST(SketchColumnTestSuite, TestClear);
FRIEND_TEST(SketchColumnTestSuite, TestClearMerge);
FRIEND_TEST(SketchColumnTestSuite, TestUpdateReallocation);
private:
  Bucket deterministic_bucket = {0, 0};
  Bucket *buckets;
  uint64_t seed;
  from_folly::RWSpinLock lock;
  uint8_t capacity;
public:
  void set_seed(uint64_t new_seed) { seed = new_seed; };
  uint64_t get_seed() const { return seed; };

  ResizeableSketchColumn(uint8_t start_capacity, uint64_t seed);
  ResizeableSketchColumn(const ResizeableSketchColumn &other);
  ResizeableSketchColumn& operator=(const ResizeableSketchColumn &other);

  ResizeableSketchColumn(ResizeableSketchColumn &&other) noexcept;
  ResizeableSketchColumn& operator=(ResizeableSketchColumn &&other) noexcept;
  ~ResizeableSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  
  void prefetch();
  
  const ColumnEntryDelta generate_entry_delta(vec_t update) const;
  void apply_entry_delta(const ColumnEntryDelta &delta);
  
  void atomic_update(const vec_t update);
  void atomic_apply_entry_delta(const ColumnEntryDelta &delta);
  void merge(ResizeableSketchColumn const& other);
  uint8_t get_depth() const;
  size_t space_usage_bytes() const;

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
  
  inline bool is_initialized() const {
    return buckets != nullptr;
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
  uint64_t get_seed() const { return seed; };

  ResizeableAlignedSketchColumn(uint8_t start_capacity, uint64_t seed);
  ResizeableAlignedSketchColumn(const ResizeableAlignedSketchColumn &other);
  ResizeableAlignedSketchColumn& operator=(const ResizeableAlignedSketchColumn &other);

  ResizeableAlignedSketchColumn(ResizeableAlignedSketchColumn &&other) noexcept;
  ResizeableAlignedSketchColumn& operator=(ResizeableAlignedSketchColumn &&other) noexcept;
  ~ResizeableAlignedSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void prefetch() {}; // TODO - implement prefetching 

  const ColumnEntryDelta generate_entry_delta(vec_t update) const;
  void apply_entry_delta(const ColumnEntryDelta &delta);

  // TODO - implement later
  void atomic_apply_entry_delta(const ColumnEntryDelta &delta) {
    this->apply_entry_delta(delta);
  }
  void atomic_update(const vec_t update) {
    // TODO - implement later
    this->update(update);
  }
  void merge(ResizeableAlignedSketchColumn const& other);
  uint8_t get_depth() const;
  size_t space_usage_bytes() const;

  [[deprecated]]
  void zero_contents() {
    clear();
  }

  inline bool is_initialized() const {
    return aligned_buckets != nullptr;
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

