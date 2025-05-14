#include "sketch/sketch_columns.h"

FixedSizeSketchColumn::FixedSizeSketchColumn(uint8_t capacity, uint16_t col_idx) :
    capacity(capacity), col_idx(col_idx) {
  buckets = new Bucket[capacity];
  std::memset(buckets, 0, capacity * sizeof(Bucket));
}

FixedSizeSketchColumn::FixedSizeSketchColumn(const FixedSizeSketchColumn &other) :
    capacity(other.capacity), col_idx(other.col_idx), deterministic_bucket(other.deterministic_bucket) {
  buckets = new Bucket[capacity];
  std::memcpy(buckets, other.buckets, capacity * sizeof(Bucket));
}

FixedSizeSketchColumn::~FixedSizeSketchColumn() {
  delete[] buckets;
}

uint8_t FixedSizeSketchColumn::get_depth() const {
  for (size_t i = capacity; i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(buckets[i - 1])) {
      return i;
    }
  }
  return 0;
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
  col_hash_t depth = Bucket_Boruvka::get_index_depth(update, seed, col_idx, capacity-1);
  assert(depth < capacity);
  buckets[depth] ^= {update, checksum};
  deterministic_bucket ^= {update, checksum};
}


ResizeableSketchColumn::ResizeableSketchColumn(uint8_t start_capacity, uint16_t col_idx) :
    capacity(start_capacity), col_idx(col_idx) {
    aligned_buckets = new Bucket[start_capacity];
    std::memset(aligned_buckets, 0, capacity * sizeof(Bucket));
}

ResizeableSketchColumn::ResizeableSketchColumn(const ResizeableSketchColumn &other) :
    capacity(other.capacity), col_idx(other.col_idx), deterministic_bucket(other.deterministic_bucket) {
  aligned_buckets = new Bucket[capacity];
  std::memcpy(aligned_buckets, other.aligned_buckets, capacity * sizeof(Bucket));
}

ResizeableSketchColumn::~ResizeableSketchColumn() {
  delete[] aligned_buckets;
}

/*
  Note this DROPS the contents if allocated down too much.
*/
void ResizeableSketchColumn::reallocate(uint8_t new_capacity) {
  // std::cout << "Reallocating from " << (int)capacity << " to " << (int)new_capacity << std::endl;
  auto new_buckets = new Bucket[new_capacity];
  likely_if (new_capacity > capacity) {
    std::memset(new_buckets + capacity, 0,
                (new_capacity - capacity) * sizeof(Bucket));
  }
  std::memcpy(new_buckets, aligned_buckets,
              std::min(capacity, new_capacity) * sizeof(Bucket));
  delete[] aligned_buckets;
  
  aligned_buckets = new_buckets;
  capacity = new_capacity;
}
void ResizeableSketchColumn::clear() {
  std::memset(aligned_buckets, 0, capacity * sizeof(Bucket));
  deterministic_bucket = {0, 0};
}

void ResizeableSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) aligned_buckets, capacity * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket, sizeof(Bucket));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
  binary_out.write((char *) &col_idx, sizeof(uint8_t));
}

SketchSample<vec_t> ResizeableSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket)) {
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }
  for (size_t i = capacity; i > 0; --i) {
    if (Bucket_Boruvka::is_good(aligned_buckets[i - 1], seed)) {
      return {aligned_buckets[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void ResizeableSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  // TODO - remove magic number
  // TODO - get_index_depth needs to be fixed. hashes need to be longer
  // than 32 bits if we're not using the deep bucket buffer idea.
  col_hash_t depth = Bucket_Boruvka::get_index_depth(update, seed, col_idx, 60);
  deterministic_bucket ^= {update, checksum};

  if (depth >= capacity) {
    size_t new_capacity = ((depth >> 2) << 2) + 4;
    reallocate(new_capacity); 
  }
  aligned_buckets[depth] ^= {update, checksum};
}

void ResizeableSketchColumn::merge(ResizeableSketchColumn &other) {
  deterministic_bucket ^= other.deterministic_bucket;
  if (other.capacity > capacity) {
    reallocate(other.capacity);
  }
  for (size_t i = 0; i < other.capacity; ++i) {
    aligned_buckets[i] ^= other.aligned_buckets[i];
  }
}

uint8_t ResizeableSketchColumn::get_depth() const {
  // TODO - maybe rely on flag vectors
  for (size_t i = capacity; i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(aligned_buckets[i - 1])) {
      return i;
    }
  }
  return 0;
}



ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(uint8_t start_capacity, uint16_t col_idx) :
    capacity(start_capacity), col_idx(col_idx) {
      
    // auto aligned_memptr = hwy::MakeUniqueAlignedArray<Bucket>(start_capacity);
    aligned_buckets = hwy::AllocateAligned<Bucket>(start_capacity);
    std::memset(aligned_buckets.get(), 0, capacity * sizeof(Bucket));
}

ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(const ResizeableAlignedSketchColumn &other) :
    capacity(other.capacity), col_idx(other.col_idx), deterministic_bucket(other.deterministic_bucket) {
  aligned_buckets = hwy::AllocateAligned<Bucket>(capacity);
  std::memcpy(aligned_buckets.get(), other.aligned_buckets.get(), capacity * sizeof(Bucket));
}

ResizeableAlignedSketchColumn::~ResizeableAlignedSketchColumn() {
}

/*
  Note this DROPS the contents if allocated down too much.
*/
void ResizeableAlignedSketchColumn::reallocate(uint8_t new_capacity) {
  auto resize_capacity = std::max(new_capacity, capacity);
  auto new_buckets = hwy::AllocateAligned<Bucket>(new_capacity);
  std::memset(new_buckets.get() + capacity, 0,
              (resize_capacity - capacity) * sizeof(Bucket));
  // old capacity:
  std::memcpy(new_buckets.get(), aligned_buckets.get(),
              capacity * sizeof(Bucket));
  aligned_buckets = std::move(new_buckets);
  capacity = new_capacity;
}

void ResizeableAlignedSketchColumn::clear() {
  std::memset(aligned_buckets.get(), 0, capacity * sizeof(Bucket));
  deterministic_bucket = {0, 0};
}

void ResizeableAlignedSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) aligned_buckets.get(), capacity * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket, sizeof(Bucket));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
  binary_out.write((char *) &col_idx, sizeof(uint8_t));
}

SketchSample<vec_t> ResizeableAlignedSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket)) {
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }
  for (size_t i = capacity; i > 0; --i) {
    if (Bucket_Boruvka::is_good(aligned_buckets[i - 1], seed)) {
      return {aligned_buckets[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void ResizeableAlignedSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  // TODO - remove magic number
  // TODO - get_index_depth needs to be fixed. hashes need to be longer
  // than 32 bits if we're not using the deep bucket buffer idea.
  col_hash_t depth = Bucket_Boruvka::get_index_depth(update, seed, col_idx, 60);
  deterministic_bucket ^= {update, checksum};

  if (depth >= capacity) {
    size_t new_capacity = ((depth >> 2) << 2) + 4;
    reallocate(new_capacity); 
  }
  aligned_buckets[depth] ^= {update, checksum};
}

void ResizeableAlignedSketchColumn::merge(ResizeableAlignedSketchColumn &other) {
  deterministic_bucket ^= other.deterministic_bucket;
  if (other.capacity > capacity) {
    reallocate(other.capacity);
  }
  uint32_t *for_vector_merge = reinterpret_cast<uint32_t*>(aligned_buckets.get());
  uint32_t *other_for_vector_merge = reinterpret_cast<uint32_t*>(other.aligned_buckets.get());
  int num_vectors = other.capacity * (sizeof(Bucket) / sizeof(uint32_t));
  hwy::HWY_NAMESPACE::simd_xor(for_vector_merge, other_for_vector_merge, num_vectors);
}

uint8_t ResizeableAlignedSketchColumn::get_depth() const {
  // TODO - maybe rely on flag vectors
  for (size_t i = capacity; i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(aligned_buckets[i - 1])) {
      return i;
    }
  }
  return 0;
}

uint64_t ResizeableSketchColumn::seed = 0;
uint64_t FixedSizeSketchColumn::seed = 0;
uint64_t ResizeableAlignedSketchColumn::seed = 0;



static_assert(SketchColumnConcept<FixedSizeSketchColumn, vec_t>,
              "FixedSizeSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableSketchColumn, vec_t>,
              "ResizeableSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableAlignedSketchColumn, vec_t>,
              "ResizeableAlignedSketchColumn does not satisfy SketchColumnConcept");