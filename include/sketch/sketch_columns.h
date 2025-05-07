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
  static uint64_t seed;

  Bucket *buckets;
  Bucket deterministic_bucket = {0, 0};
  uint8_t capacity;
  uint8_t col_idx; // determines column seeding
public:
  static void set_seed(uint64_t new_seed) {
    seed = new_seed;
  };  
  static const uint64_t get_seed() {
    return seed;
  };

  FixedSizeSketchColumn(uint8_t capacity, uint8_t col_idx);
  ~FixedSizeSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void merge(FixedSizeSketchColumn &other);
  uint8_t get_depth() const;
  void serialize(std::ostream &binary_out) const;
};

FixedSizeSketchColumn::FixedSizeSketchColumn(uint8_t capacity, uint8_t col_idx) :
    capacity(capacity), col_idx(col_idx) {
  buckets = new Bucket[capacity];
  // std::memset(buckets, 0, capacity * sizeof(Bucket));
}

FixedSizeSketchColumn::~FixedSizeSketchColumn() {
  delete[] buckets;
}

uint8_t FixedSizeSketchColumn::get_depth() const {
  for (size_t i = capacity; i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(buckets[i - 1])) {
      return i - 1;
    }
  }
}

// TODO - implement actual deserialization
void FixedSizeSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) buckets, capacity * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket, sizeof(Bucket));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
  binary_out.write((char *) &col_idx, sizeof(uint8_t));
}

SketchSample<vec_t> FixedSizeSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket)) {
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }
  for (size_t i = 0; i < capacity; ++i) {
    if (Bucket_Boruvka::is_good(buckets[i], seed)) {
      return {buckets[i].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void FixedSizeSketchColumn::clear() {
  std::memset(buckets, 0, capacity * sizeof(Bucket));
  deterministic_bucket = {0, 0};
}

void FixedSizeSketchColumn::merge(FixedSizeSketchColumn &other) {
  for (size_t i = 0; i < capacity; ++i) {
    buckets[i] ^= other.buckets[i];
  }
  deterministic_bucket ^= other.deterministic_bucket;
}

void FixedSizeSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth(update, seed, col_idx, capacity);
  buckets[depth] ^= {update, checksum};
  deterministic_bucket ^= {update, checksum};
}



class ResizeableSketchColumn {
private:
  static uint64_t seed;

  hwy::AlignedFreeUniquePtr<Bucket[]> aligned_buckets;
  Bucket deterministic_bucket = {0, 0};
  uint8_t capacity;
  uint8_t col_idx; // determines column seeding
public:
  static void set_seed(uint64_t new_seed) { seed = new_seed; };
  static const uint64_t get_seed() { return seed; };

  ResizeableSketchColumn(uint8_t start_capacity, uint8_t col_idx);
  ~ResizeableSketchColumn();
  SketchSample<vec_t> sample() const;
  void clear();
  void update(const vec_t update);
  void merge(ResizeableSketchColumn &other);
  uint8_t get_depth() const;
  void serialize(std::ostream &binary_out) const;
private:
  void reallocate(uint8_t new_capacity);
};


ResizeableSketchColumn::ResizeableSketchColumn(uint8_t start_capacity, uint8_t col_idx) :
    capacity(start_capacity), col_idx(col_idx) {
      
    // auto aligned_memptr = hwy::MakeUniqueAlignedArray<Bucket>(start_capacity);
    aligned_buckets = hwy::AllocateAligned<Bucket>(start_capacity);
    std::memset(aligned_buckets.get(), 0, capacity * sizeof(Bucket));
}

/*
  Note this DROPS the contents if allocated down too much.
*/
void ResizeableSketchColumn::reallocate(uint8_t new_capacity) {
  auto resize_capacity = std::max(new_capacity, capacity);
  auto new_buckets = hwy::AllocateAligned<Bucket>(new_capacity);
  std::memset(new_buckets.get() + capacity, 0,
              (resize_capacity - capacity) * sizeof(Bucket));
  std::memcpy(new_buckets.get(), aligned_buckets.get(),
              resize_capacity * sizeof(Bucket));
  aligned_buckets = std::move(new_buckets);
  capacity = new_capacity;
}

void ResizeableSketchColumn::clear() {
  std::memset(aligned_buckets.get(), 0, capacity * sizeof(Bucket));
  deterministic_bucket = {0, 0};
}

void ResizeableSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) aligned_buckets.get(), capacity * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket, sizeof(Bucket));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
  binary_out.write((char *) &col_idx, sizeof(uint8_t));
}

void ResizeableSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth(update, seed, col_idx, capacity);
  deterministic_bucket ^= {update, checksum};

  if (depth >= capacity) {
    reallocate(depth + 4);
  }
  aligned_buckets[depth] ^= {update, checksum};
}

void ResizeableSketchColumn::merge(ResizeableSketchColumn &other) {
  deterministic_bucket ^= other.deterministic_bucket;
  if (other.capacity > capacity) {
    reallocate(other.capacity);
  }
  // auto for_vector_merge = hwy::Rebind<Bucket, uint32_t(aligned_buckets.get(), capacity);
  uint32_t *for_vector_merge = reinterpret_cast<uint32_t*>(aligned_buckets.get());
  uint32_t *other_for_vector_merge = reinterpret_cast<uint32_t*>(other.aligned_buckets.get());
  int num_vectors = other.capacity * (sizeof(Bucket) / sizeof(uint32_t));
  hwy::HWY_NAMESPACE::simd_xor(for_vector_merge, other_for_vector_merge, num_vectors);
}

uint8_t ResizeableSketchColumn::get_depth() const {
  // TODO - maybe rely on flag vectors
  for (size_t i = capacity; i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(aligned_buckets[i - 1])) {
      return i - 1;
    }
  }
}


static_assert(SketchColumnConcept<FixedSizeSketchColumn, vec_t>,
              "FixedSizeSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableSketchColumn, vec_t>,
              "ResizeableSketchColumn does not satisfy SketchColumnConcept");