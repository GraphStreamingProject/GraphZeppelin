#include "sketch/sketch_columns.h"

FixedSizeSketchColumn::FixedSizeSketchColumn(uint8_t capacity, uint64_t seed) :
    capacity(capacity), seed(seed) {
  likely_if (capacity > 0) {
    buckets = new Bucket[capacity];
    std::memset(buckets, 0, capacity * sizeof(Bucket));
  }
  else {
    buckets = nullptr;
  }
}

FixedSizeSketchColumn::FixedSizeSketchColumn(const FixedSizeSketchColumn &other) :
    capacity(other.capacity), seed(other.seed), deterministic_bucket(other.deterministic_bucket) {
  likely_if (capacity > 0) {
    buckets = new Bucket[capacity];
    std::memcpy(buckets, other.buckets, capacity * sizeof(Bucket));
  }
  else {
    buckets = nullptr;
  }
}

FixedSizeSketchColumn::FixedSizeSketchColumn(FixedSizeSketchColumn &&other) noexcept :
    capacity(other.capacity), seed(other.seed), deterministic_bucket(other.deterministic_bucket) {
      buckets = other.buckets;
      other.buckets = nullptr;
      other.capacity = 0;
}

FixedSizeSketchColumn& FixedSizeSketchColumn::operator=(FixedSizeSketchColumn &&other) noexcept {
  if (this != &other) {
    delete[] buckets;
    capacity = other.capacity;
    seed = other.seed;
    deterministic_bucket = other.deterministic_bucket;
    
    buckets = other.buckets;
    other.buckets = nullptr;
    other.capacity = 0;
  }
  return *this;
}
FixedSizeSketchColumn& FixedSizeSketchColumn::operator=(const FixedSizeSketchColumn &other) {
  if (this != &other) {
    delete[] buckets;
    capacity = other.capacity;
    seed = other.seed;
    deterministic_bucket = other.deterministic_bucket;
    
    buckets = new Bucket[capacity];
    std::memcpy(buckets, other.buckets, capacity * sizeof(Bucket));
  }
  // TODO - an else case?
  return *this;
}

FixedSizeSketchColumn::~FixedSizeSketchColumn() {
  // note nullptr is safe to delete
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
size_t FixedSizeSketchColumn::space_usage_bytes() const {
  return sizeof(FixedSizeSketchColumn) + (capacity * sizeof(Bucket));
}

// TODO - implement actual deserialization
void FixedSizeSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) buckets, capacity * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket, sizeof(Bucket));
  binary_out.write((char *) &seed, sizeof(uint64_t));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
}

// TODO: track deepest nonzero bucket after every update, use this to optimize bottom-up query search for good buckets
SketchSample<vec_t> FixedSizeSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket)) {
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }
  for (size_t i = capacity; i > 0; --i) {
    if (Bucket_Boruvka::is_good(buckets[i - 1], seed)) {
      return {buckets[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void FixedSizeSketchColumn::clear() {
  std::memset(buckets, 0, capacity * sizeof(Bucket));
  deterministic_bucket = {0, 0};
}

void FixedSizeSketchColumn::prefetch() {
  constexpr size_t cache_line_size = 64; // assuming 64 byte cache lines
  size_t num_lines = (capacity * sizeof(Bucket) + cache_line_size - 1) / cache_line_size;
  for (size_t i = 0; i < num_lines; ++i) {
    _mm_prefetch(reinterpret_cast<const char*>(buckets) + i * cache_line_size, _MM_HINT_T0);
  }
}

void FixedSizeSketchColumn::merge(FixedSizeSketchColumn const& other) {
  for (size_t i = 0; i < capacity; ++i) {
    buckets[i] ^= other.buckets[i];
  }
  deterministic_bucket ^= other.deterministic_bucket;
}

void FixedSizeSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, capacity-1);
  // assert(depth < capacity);
  buckets[depth] ^= {update, checksum};
  deterministic_bucket ^= {update, checksum};
}

const ColumnEntryDelta FixedSizeSketchColumn::generate_entry_delta(vec_t update) const {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, capacity-1);
  return {Bucket{update, checksum}, static_cast<uint16_t>(depth)};
}

void FixedSizeSketchColumn::apply_entry_delta(const ColumnEntryDelta &delta) {
  assert(delta.depth < capacity);
  buckets[delta.depth] ^= delta.bucket;
  deterministic_bucket ^= delta.bucket;
}

void FixedSizeSketchColumn::atomic_update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, capacity-1);
  
  std::atomic_ref<vec_t> det_alpha(deterministic_bucket.alpha);
  std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket.gamma);
  
  std::atomic_ref<vec_t> bucket_alpha(buckets[depth].alpha);
  std::atomic_ref<vec_hash_t> bucket_gamma(buckets[depth].gamma);
  
  det_alpha.fetch_xor(update, std::memory_order_relaxed);
  det_gamma.fetch_xor(checksum, std::memory_order_relaxed);
  bucket_alpha.fetch_xor(update, std::memory_order_relaxed);
  bucket_gamma.fetch_xor(checksum, std::memory_order_relaxed);
  // todo - gccc intrinsics?

}
void FixedSizeSketchColumn::atomic_apply_entry_delta(const ColumnEntryDelta &delta) {
  assert(delta.depth < capacity);
  
  std::atomic_ref<vec_t> det_alpha(deterministic_bucket.alpha);
  std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket.gamma);
  
  std::atomic_ref<vec_t> bucket_alpha(buckets[delta.depth].alpha);
  std::atomic_ref<vec_hash_t> bucket_gamma(buckets[delta.depth].gamma);
  
  det_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
  det_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
  bucket_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
  bucket_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
  // todo - gccc intrinsics?
}

ResizeableSketchColumn::ResizeableSketchColumn(uint8_t start_capacity,
                                               uint64_t seed)
    : capacity(start_capacity), seed(seed) {
  likely_if(capacity > 0) {
    buckets = new Bucket[start_capacity];
    std::memset(buckets, 0, capacity * sizeof(Bucket));
  }
  else {
    buckets = nullptr;
  }
}

ResizeableSketchColumn::ResizeableSketchColumn(const ResizeableSketchColumn &other) :
    capacity(other.capacity), seed(other.seed), deterministic_bucket(other.deterministic_bucket) {
  likely_if(capacity > 0) {
    buckets = new Bucket[capacity];
    std::memcpy(buckets, other.buckets, capacity * sizeof(Bucket));
  }
  else {
    buckets = nullptr;
  }
}

ResizeableSketchColumn::ResizeableSketchColumn(ResizeableSketchColumn &&other) noexcept :
    capacity(other.capacity), seed(other.seed), deterministic_bucket(other.deterministic_bucket) {
    // move constructor
    buckets = other.buckets;
    other.buckets = nullptr;
    other.capacity = 0;
}

ResizeableSketchColumn& ResizeableSketchColumn::operator=(ResizeableSketchColumn &&other) noexcept {
  if (this != &other) {
    delete[] buckets;
    capacity = other.capacity;
    seed = other.seed;
    deterministic_bucket = other.deterministic_bucket;
    
    buckets = other.buckets;
    other.buckets = nullptr;
    other.capacity = 0;
  }
  return *this;
}
ResizeableSketchColumn& ResizeableSketchColumn::operator=(const ResizeableSketchColumn &other) {
  if (this != &other) {
    delete[] buckets;
    capacity = other.capacity;
    seed = other.seed;
    deterministic_bucket = other.deterministic_bucket;
    
    buckets = new Bucket[capacity];
    std::memcpy(buckets, other.buckets, capacity * sizeof(Bucket));
  }
  // TODO - an else case?
  return *this;
}

ResizeableSketchColumn::~ResizeableSketchColumn() {
  delete[] buckets;
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
  std::memcpy(new_buckets, buckets,
              std::min(capacity, new_capacity) * sizeof(Bucket));
  delete[] buckets;
  
  buckets = new_buckets;
  capacity = new_capacity;
}
void ResizeableSketchColumn::clear() {
  std::memset(buckets, 0, capacity * sizeof(Bucket));
  deterministic_bucket = {0, 0};
}

void ResizeableSketchColumn::prefetch() {
  constexpr size_t cache_line = 64; // bytes
  size_t num_lines = (capacity * sizeof(Bucket) + cache_line - 1) / cache_line;
  // prefetch all buckets
  for (size_t i = 0; i < num_lines; ++i) {
    _mm_prefetch(reinterpret_cast<const char*>(buckets) + (i * cache_line), _MM_HINT_T0);
  }
}

void ResizeableSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) buckets, capacity * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket, sizeof(Bucket));
  binary_out.write((char *) &seed, sizeof(uint64_t));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
}

SketchSample<vec_t> ResizeableSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket)) {
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }
  for (size_t i = capacity; i > 0; --i) {
    if (Bucket_Boruvka::is_good(buckets[i - 1], seed)) {
      return {buckets[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void ResizeableSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  // TODO - remove magic number
  // TODO - get_index_depth needs to be fixed. hashes need to be longer
  // than 32 bits if we're not using the deep bucket buffer idea.
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  deterministic_bucket ^= {update, checksum};

  if (depth >= capacity) {
    size_t new_capacity = ((depth >> 2) << 2) + 4;
    reallocate(new_capacity); 
  }
  buckets[depth] ^= {update, checksum};
}

const ColumnEntryDelta ResizeableSketchColumn::generate_entry_delta(vec_t update) const {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  return {Bucket{update, checksum}, static_cast<uint16_t>(depth)};
}

void ResizeableSketchColumn::apply_entry_delta(const ColumnEntryDelta &delta) {
  deterministic_bucket ^= delta.bucket;

  if (delta.depth >= capacity) {
    size_t new_capacity = ((delta.depth >> 2) << 2) + 4;
    reallocate(new_capacity);
  }
  buckets[delta.depth] ^= delta.bucket;
}

void ResizeableSketchColumn::atomic_update(const vec_t update) {
  // TODO - there's code duplication with apply entry delta.
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  

  // grab reader lock
  this->lock.lock_shared();
  if (depth < capacity) {
    // can atomically update as normal
    std::atomic_ref<vec_t> det_alpha(deterministic_bucket.alpha);
    std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket.gamma);
    std::atomic_ref<vec_t> bucket_alpha(buckets[depth].alpha);
    std::atomic_ref<vec_hash_t> bucket_gamma(buckets[depth].gamma);
    det_alpha.fetch_xor(update, std::memory_order_relaxed);
    det_gamma.fetch_xor(checksum, std::memory_order_relaxed);
    bucket_alpha.fetch_xor(update, std::memory_order_relaxed);
    bucket_gamma.fetch_xor(checksum, std::memory_order_relaxed);
    this->lock.unlock_shared();
  } else {
    // release the reader lock
    this->lock.unlock_shared();
    // grab writer lock
    this->lock.lock();
    // note: the alllocation may have shrunk OR grown.
    // so we need to account for that
    size_t desired_capacity = ((depth >> 2) << 2) + 4;
    desired_capacity = std::max(desired_capacity, static_cast<size_t>(capacity));
    if (desired_capacity != capacity) {
      reallocate(desired_capacity);
    }
    // now we can update the buckets (non-atomically)
    deterministic_bucket ^= {update, checksum};
    buckets[depth] ^= {update, checksum};
    this->lock.unlock();
  }
  
}
void ResizeableSketchColumn::atomic_apply_entry_delta(const ColumnEntryDelta &delta) {
  
  // grab reader lock
  this->lock.lock_shared();
  if (delta.depth < capacity) {
    // can atomically update as normal
    std::atomic_ref<vec_t> det_alpha(deterministic_bucket.alpha);
    std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket.gamma);
    std::atomic_ref<vec_t> bucket_alpha(buckets[delta.depth].alpha);
    std::atomic_ref<vec_hash_t> bucket_gamma(buckets[delta.depth].gamma);
    det_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
    det_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
    bucket_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
    bucket_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
    this->lock.unlock_shared();
  } else {
    // release the reader lock
    this->lock.unlock_shared();
    // grab writer lock
    this->lock.lock();
    // note: the alllocation may have shrunk OR grown.
    // so we need to account for that
    size_t desired_capacity = ((delta.depth >> 2) << 2) + 4;
    desired_capacity = std::max(desired_capacity, static_cast<size_t>(capacity));
    if (desired_capacity != capacity) {
      reallocate(desired_capacity);
    }
    // now we can update the buckets (non-atomically)
    deterministic_bucket ^= delta.bucket;
    buckets[delta.depth] ^= delta.bucket;
    this->lock.unlock();
  }
  
}

void ResizeableSketchColumn::merge(ResizeableSketchColumn const& other) {
  deterministic_bucket ^= other.deterministic_bucket;
  if (other.capacity > capacity) {
    reallocate(other.capacity);
  }
  for (size_t i = 0; i < other.capacity; ++i) {
    buckets[i] ^= other.buckets[i];
  }
}

uint8_t ResizeableSketchColumn::get_depth() const {
  // TODO - maybe rely on flag vectors
  for (size_t i = capacity; i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(buckets[i - 1])) {
      return i;
    }
  }
  return 0;
}
size_t ResizeableSketchColumn::space_usage_bytes() const {
  return sizeof(ResizeableSketchColumn) + (this->capacity * sizeof(Bucket));
}



ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(uint8_t start_capacity, uint64_t seed) :
    capacity(start_capacity), seed(seed) {
      
    // auto aligned_memptr = hwy::MakeUniqueAlignedArray<Bucket>(start_capacity);
    aligned_buckets = hwy::AllocateAligned<Bucket>(start_capacity);
    std::memset(aligned_buckets.get(), 0, capacity * sizeof(Bucket));
}

ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(const ResizeableAlignedSketchColumn &other) :
    capacity(other.capacity), seed(other.seed), deterministic_bucket(other.deterministic_bucket) {
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
  binary_out.write((char *) &seed, sizeof(uint64_t));
  binary_out.write((char *) &capacity, sizeof(uint8_t));
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
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  deterministic_bucket ^= {update, checksum};

  if (depth >= capacity) {
    size_t new_capacity = ((depth >> 2) << 2) + 4;
    reallocate(new_capacity); 
  }
  aligned_buckets[depth] ^= {update, checksum};
}

const ColumnEntryDelta ResizeableAlignedSketchColumn::generate_entry_delta(vec_t update) const {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  return {Bucket{update, checksum}, static_cast<uint16_t>(depth)};
}

void ResizeableAlignedSketchColumn::apply_entry_delta(const ColumnEntryDelta &delta) {
  assert(delta.depth < capacity);
  deterministic_bucket ^= delta.bucket;

  if (delta.depth >= capacity) {
    size_t new_capacity = ((delta.depth >> 2) << 2) + 4;
    reallocate(new_capacity);
  }
  aligned_buckets[delta.depth] ^= delta.bucket;
}

void ResizeableAlignedSketchColumn::merge(ResizeableAlignedSketchColumn const& other) {
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
size_t ResizeableAlignedSketchColumn::space_usage_bytes() const{
  return sizeof(ResizeableAlignedSketchColumn) + (this->capacity * sizeof(Bucket));
}


static_assert(SketchColumnConcept<FixedSizeSketchColumn, vec_t>,
              "FixedSizeSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableSketchColumn, vec_t>,
              "ResizeableSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableAlignedSketchColumn, vec_t>,
              "ResizeableAlignedSketchColumn does not satisfy SketchColumnConcept");