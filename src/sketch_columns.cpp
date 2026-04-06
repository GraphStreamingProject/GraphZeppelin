#include "sketch/sketch_columns.h"

namespace {
constexpr uintptr_t kCapacityMask = 0xFFu;
constexpr std::align_val_t kTaggedAlignment = std::align_val_t(256);

Bucket *allocate_bucket_block_impl(uint8_t capacity) {
  const size_t count = static_cast<size_t>(capacity) + 1;
  auto *raw = static_cast<Bucket*>(::operator new[](count * sizeof(Bucket), kTaggedAlignment));
  std::memset(raw, 0, count * sizeof(Bucket));
  return raw;
}

void free_bucket_block_impl(Bucket *raw) {
  ::operator delete[](raw, kTaggedAlignment);
}
}  // namespace

void FixedSizeSketchColumn::set_tagged_buckets(Bucket *raw_buckets, uint8_t capacity) {
  assert((reinterpret_cast<uintptr_t>(raw_buckets) & kCapacityMask) == 0);
  buckets_tagged = reinterpret_cast<uintptr_t>(raw_buckets) | capacity;
}

Bucket *FixedSizeSketchColumn::allocate_bucket_block(uint8_t capacity) {
  return allocate_bucket_block_impl(capacity);
}

void FixedSizeSketchColumn::free_bucket_block(Bucket *raw_buckets) {
  if (raw_buckets != nullptr) {
    free_bucket_block_impl(raw_buckets);
  }
}

FixedSizeSketchColumn::FixedSizeSketchColumn(uint8_t capacity, uint64_t seed)
    : seed(seed) {
  set_tagged_buckets(allocate_bucket_block(capacity), capacity);
}

FixedSizeSketchColumn::FixedSizeSketchColumn(const FixedSizeSketchColumn &other)
    : seed(other.seed) {
  set_tagged_buckets(allocate_bucket_block(other.capacity()), other.capacity());
  std::memcpy(buckets_raw(), other.buckets_raw(),
              (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

FixedSizeSketchColumn::FixedSizeSketchColumn(FixedSizeSketchColumn &&other) noexcept
    : buckets_tagged(other.buckets_tagged), seed(other.seed) {
  other.buckets_tagged = 0;
}

FixedSizeSketchColumn& FixedSizeSketchColumn::operator=(FixedSizeSketchColumn &&other) noexcept {
  if (this != &other) {
    free_bucket_block(buckets_raw());
    buckets_tagged = other.buckets_tagged;
    seed = other.seed;
    other.buckets_tagged = 0;
  }
  return *this;
}

FixedSizeSketchColumn& FixedSizeSketchColumn::operator=(const FixedSizeSketchColumn &other) {
  if (this != &other) {
    free_bucket_block(buckets_raw());
    seed = other.seed;
    set_tagged_buckets(allocate_bucket_block(other.capacity()), other.capacity());
    std::memcpy(buckets_raw(), other.buckets_raw(),
                (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
  }
  return *this;
}

FixedSizeSketchColumn::~FixedSizeSketchColumn() {
  free_bucket_block(buckets_raw());
}

uint8_t FixedSizeSketchColumn::get_depth() const {
  for (size_t i = capacity(); i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(buckets()[i - 1])) {
      return i;
    }
  }
  return 0;
}

size_t FixedSizeSketchColumn::space_usage_bytes() const {
  return sizeof(FixedSizeSketchColumn) + ((static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

void FixedSizeSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) buckets(), capacity() * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket_ref(), sizeof(Bucket));
  binary_out.write((char *) &seed, sizeof(uint64_t));
  const uint8_t packed_capacity = capacity();
  binary_out.write((char *) &packed_capacity, sizeof(uint8_t));
}

SketchSample<vec_t> FixedSizeSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket_ref())) {
    return {0, ZERO};
  }
  for (size_t i = capacity(); i > 0; --i) {
    if (Bucket_Boruvka::is_good(buckets()[i - 1], seed)) {
      return {buckets()[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void FixedSizeSketchColumn::clear() {
  std::memset(buckets_raw(), 0, (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

void FixedSizeSketchColumn::prefetch() {
  constexpr size_t cache_line_size = 64;
  size_t num_lines = (capacity() * sizeof(Bucket) + cache_line_size - 1) / cache_line_size;
  for (size_t i = 0; i < num_lines; ++i) {
    _mm_prefetch(reinterpret_cast<const char*>(buckets()) + i * cache_line_size, _MM_HINT_T0);
  }
}

void FixedSizeSketchColumn::merge(FixedSizeSketchColumn const& other) {
  for (size_t i = 0; i < capacity(); ++i) {
    buckets()[i] ^= other.buckets()[i];
  }
  deterministic_bucket_ref() ^= other.deterministic_bucket_ref();
}

void FixedSizeSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, capacity() - 1);
  buckets()[depth] ^= {update, checksum};
  deterministic_bucket_ref() ^= {update, checksum};
}

const ColumnEntryDelta FixedSizeSketchColumn::generate_entry_delta(vec_t update) const {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, capacity() - 1);
  return {Bucket{update, checksum}, static_cast<uint16_t>(depth)};
}

void FixedSizeSketchColumn::apply_entry_delta(const ColumnEntryDelta &delta) {
  assert(delta.depth < capacity());
  buckets()[delta.depth] ^= delta.bucket;
  deterministic_bucket_ref() ^= delta.bucket;
}

void FixedSizeSketchColumn::atomic_update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, capacity() - 1);

  std::atomic_ref<vec_t> det_alpha(deterministic_bucket_ref().alpha);
  std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket_ref().gamma);

  std::atomic_ref<vec_t> bucket_alpha(buckets()[depth].alpha);
  std::atomic_ref<vec_hash_t> bucket_gamma(buckets()[depth].gamma);

  det_alpha.fetch_xor(update, std::memory_order_relaxed);
  det_gamma.fetch_xor(checksum, std::memory_order_relaxed);
  bucket_alpha.fetch_xor(update, std::memory_order_relaxed);
  bucket_gamma.fetch_xor(checksum, std::memory_order_relaxed);
}

void FixedSizeSketchColumn::atomic_apply_entry_delta(const ColumnEntryDelta &delta) {
  assert(delta.depth < capacity());

  std::atomic_ref<vec_t> det_alpha(deterministic_bucket_ref().alpha);
  std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket_ref().gamma);

  std::atomic_ref<vec_t> bucket_alpha(buckets()[delta.depth].alpha);
  std::atomic_ref<vec_hash_t> bucket_gamma(buckets()[delta.depth].gamma);

  det_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
  det_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
  bucket_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
  bucket_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
}

void ResizeableSketchColumn::set_tagged_buckets(Bucket *raw_buckets, uint8_t capacity) {
  assert((reinterpret_cast<uintptr_t>(raw_buckets) & kCapacityMask) == 0);
  buckets_tagged = reinterpret_cast<uintptr_t>(raw_buckets) | capacity;
}

Bucket *ResizeableSketchColumn::allocate_bucket_block(uint8_t capacity) {
  return allocate_bucket_block_impl(capacity);
}

void ResizeableSketchColumn::free_bucket_block(Bucket *raw_buckets) {
  if (raw_buckets != nullptr) {
    free_bucket_block_impl(raw_buckets);
  }
}

ResizeableSketchColumn::ResizeableSketchColumn(uint8_t start_capacity,
                                               uint64_t seed)
    : seed(seed) {
  set_tagged_buckets(allocate_bucket_block(start_capacity), start_capacity);
}

ResizeableSketchColumn::ResizeableSketchColumn(const ResizeableSketchColumn &other)
    : seed(other.seed) {
  set_tagged_buckets(allocate_bucket_block(other.capacity()), other.capacity());
  std::memcpy(buckets_raw(), other.buckets_raw(),
              (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

ResizeableSketchColumn::ResizeableSketchColumn(ResizeableSketchColumn &&other) noexcept
    : buckets_tagged(other.buckets_tagged), seed(other.seed) {
  other.buckets_tagged = 0;
}

ResizeableSketchColumn& ResizeableSketchColumn::operator=(ResizeableSketchColumn &&other) noexcept {
  if (this != &other) {
    free_bucket_block(buckets_raw());
    buckets_tagged = other.buckets_tagged;
    seed = other.seed;
    other.buckets_tagged = 0;
  }
  return *this;
}

ResizeableSketchColumn& ResizeableSketchColumn::operator=(const ResizeableSketchColumn &other) {
  if (this != &other) {
    free_bucket_block(buckets_raw());
    seed = other.seed;
    set_tagged_buckets(allocate_bucket_block(other.capacity()), other.capacity());
    std::memcpy(buckets_raw(), other.buckets_raw(),
                (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
  }
  return *this;
}

ResizeableSketchColumn::~ResizeableSketchColumn() {
  free_bucket_block(buckets_raw());
}

void ResizeableSketchColumn::reallocate(uint8_t new_capacity) {
  Bucket *new_block = allocate_bucket_block(new_capacity);
  Bucket *old_block = buckets_raw();
  const size_t copy_capacity = std::min(capacity(), new_capacity);
  std::memcpy(new_block, old_block, (copy_capacity + 1) * sizeof(Bucket));
  free_bucket_block(old_block);
  set_tagged_buckets(new_block, new_capacity);
}

void ResizeableSketchColumn::clear() {
  std::memset(buckets_raw(), 0, (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

void ResizeableSketchColumn::prefetch() {
  constexpr size_t cache_line = 64;
  size_t num_lines = (capacity() * sizeof(Bucket) + cache_line - 1) / cache_line;
  for (size_t i = 0; i < num_lines; ++i) {
    _mm_prefetch(reinterpret_cast<const char*>(buckets()) + (i * cache_line), _MM_HINT_T0);
  }
}

void ResizeableSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) buckets(), capacity() * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket_ref(), sizeof(Bucket));
  binary_out.write((char *) &seed, sizeof(uint64_t));
  const uint8_t packed_capacity = capacity();
  binary_out.write((char *) &packed_capacity, sizeof(uint8_t));
}

SketchSample<vec_t> ResizeableSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket_ref())) {
    return {0, ZERO};
  }
  for (size_t i = capacity(); i > 0; --i) {
    if (Bucket_Boruvka::is_good(buckets()[i - 1], seed)) {
      return {buckets()[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void ResizeableSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  deterministic_bucket_ref() ^= {update, checksum};

  if (depth >= capacity()) {
    reallocate(static_cast<uint8_t>(depth + 1));
  }
  buckets()[depth] ^= {update, checksum};
}

const ColumnEntryDelta ResizeableSketchColumn::generate_entry_delta(vec_t update) const {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  return {Bucket{update, checksum}, static_cast<uint16_t>(depth)};
}

void ResizeableSketchColumn::apply_entry_delta(const ColumnEntryDelta &delta) {
  deterministic_bucket_ref() ^= delta.bucket;

  if (delta.depth >= capacity()) {
    reallocate(static_cast<uint8_t>(delta.depth + 1));
  }
  buckets()[delta.depth] ^= delta.bucket;
}

void ResizeableSketchColumn::atomic_update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);

  this->lock.lock_shared();
  if (depth < capacity()) {
    std::atomic_ref<vec_t> det_alpha(deterministic_bucket_ref().alpha);
    std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket_ref().gamma);
    std::atomic_ref<vec_t> bucket_alpha(buckets()[depth].alpha);
    std::atomic_ref<vec_hash_t> bucket_gamma(buckets()[depth].gamma);
    det_alpha.fetch_xor(update, std::memory_order_relaxed);
    det_gamma.fetch_xor(checksum, std::memory_order_relaxed);
    bucket_alpha.fetch_xor(update, std::memory_order_relaxed);
    bucket_gamma.fetch_xor(checksum, std::memory_order_relaxed);
    this->lock.unlock_shared();
  } else {
    this->lock.unlock_shared();
    this->lock.lock();
    size_t desired_capacity = std::max(static_cast<size_t>(depth + 1), static_cast<size_t>(capacity()));
    if (desired_capacity != capacity()) {
      reallocate(static_cast<uint8_t>(desired_capacity));
    }
    deterministic_bucket_ref() ^= {update, checksum};
    buckets()[depth] ^= {update, checksum};
    this->lock.unlock();
  }
}

void ResizeableSketchColumn::atomic_apply_entry_delta(const ColumnEntryDelta &delta) {
  this->lock.lock_shared();
  if (delta.depth < capacity()) {
    std::atomic_ref<vec_t> det_alpha(deterministic_bucket_ref().alpha);
    std::atomic_ref<vec_hash_t> det_gamma(deterministic_bucket_ref().gamma);
    std::atomic_ref<vec_t> bucket_alpha(buckets()[delta.depth].alpha);
    std::atomic_ref<vec_hash_t> bucket_gamma(buckets()[delta.depth].gamma);
    det_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
    det_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
    bucket_alpha.fetch_xor(delta.bucket.alpha, std::memory_order_relaxed);
    bucket_gamma.fetch_xor(delta.bucket.gamma, std::memory_order_relaxed);
    this->lock.unlock_shared();
  } else {
    this->lock.unlock_shared();
    this->lock.lock();
    size_t desired_capacity = std::max(static_cast<size_t>(delta.depth + 1), static_cast<size_t>(capacity()));
    if (desired_capacity != capacity()) {
      reallocate(static_cast<uint8_t>(desired_capacity));
    }
    deterministic_bucket_ref() ^= delta.bucket;
    buckets()[delta.depth] ^= delta.bucket;
    this->lock.unlock();
  }
}

void ResizeableSketchColumn::merge(ResizeableSketchColumn const& other) {
  deterministic_bucket_ref() ^= other.deterministic_bucket_ref();
  if (other.capacity() > capacity()) {
    reallocate(other.capacity());
  }
  for (size_t i = 0; i < other.capacity(); ++i) {
    buckets()[i] ^= other.buckets()[i];
  }

  constexpr size_t merge_slack = 1;
  size_t depth = get_depth();
  size_t target_capacity = std::max<size_t>(1, depth + merge_slack);
  if (target_capacity < capacity()) {
    reallocate(static_cast<uint8_t>(target_capacity));
  }
}

uint8_t ResizeableSketchColumn::get_depth() const {
  for (size_t i = capacity(); i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(buckets()[i - 1])) {
      return i;
    }
  }
  return 0;
}

size_t ResizeableSketchColumn::space_usage_bytes() const {
  return sizeof(ResizeableSketchColumn) + ((static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

void ResizeableAlignedSketchColumn::set_tagged_buckets(Bucket *raw_buckets, uint8_t capacity) {
  assert((reinterpret_cast<uintptr_t>(raw_buckets) & kCapacityMask) == 0);
  buckets_tagged = reinterpret_cast<uintptr_t>(raw_buckets) | capacity;
}

Bucket *ResizeableAlignedSketchColumn::allocate_bucket_block(uint8_t capacity) {
  return allocate_bucket_block_impl(capacity);
}

void ResizeableAlignedSketchColumn::free_bucket_block(Bucket *raw_buckets) {
  if (raw_buckets != nullptr) {
    free_bucket_block_impl(raw_buckets);
  }
}

ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(uint8_t start_capacity, uint64_t seed)
    : seed(seed) {
  set_tagged_buckets(allocate_bucket_block(start_capacity), start_capacity);
}

ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(const ResizeableAlignedSketchColumn &other)
    : seed(other.seed) {
  set_tagged_buckets(allocate_bucket_block(other.capacity()), other.capacity());
  std::memcpy(buckets_raw(), other.buckets_raw(),
              (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

ResizeableAlignedSketchColumn::ResizeableAlignedSketchColumn(ResizeableAlignedSketchColumn &&other) noexcept
    : buckets_tagged(other.buckets_tagged), seed(other.seed) {
  other.buckets_tagged = 0;
}

ResizeableAlignedSketchColumn& ResizeableAlignedSketchColumn::operator=(ResizeableAlignedSketchColumn &&other) noexcept {
  if (this != &other) {
    free_bucket_block(buckets_raw());
    buckets_tagged = other.buckets_tagged;
    seed = other.seed;
    other.buckets_tagged = 0;
  }
  return *this;
}

ResizeableAlignedSketchColumn& ResizeableAlignedSketchColumn::operator=(const ResizeableAlignedSketchColumn &other) {
  if (this != &other) {
    free_bucket_block(buckets_raw());
    seed = other.seed;
    set_tagged_buckets(allocate_bucket_block(other.capacity()), other.capacity());
    std::memcpy(buckets_raw(), other.buckets_raw(),
                (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
  }
  return *this;
}

ResizeableAlignedSketchColumn::~ResizeableAlignedSketchColumn() {
  free_bucket_block(buckets_raw());
}

void ResizeableAlignedSketchColumn::reallocate(uint8_t new_capacity) {
  Bucket *new_block = allocate_bucket_block(new_capacity);
  Bucket *old_block = buckets_raw();
  const size_t copy_capacity = std::min(capacity(), new_capacity);
  std::memcpy(new_block, old_block, (copy_capacity + 1) * sizeof(Bucket));
  free_bucket_block(old_block);
  set_tagged_buckets(new_block, new_capacity);
}

void ResizeableAlignedSketchColumn::clear() {
  std::memset(buckets_raw(), 0, (static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

void ResizeableAlignedSketchColumn::serialize(std::ostream &binary_out) const {
  binary_out.write((char *) buckets(), capacity() * sizeof(Bucket));
  binary_out.write((char *) &deterministic_bucket_ref(), sizeof(Bucket));
  binary_out.write((char *) &seed, sizeof(uint64_t));
  const uint8_t packed_capacity = capacity();
  binary_out.write((char *) &packed_capacity, sizeof(uint8_t));
}

SketchSample<vec_t> ResizeableAlignedSketchColumn::sample() const {
  if (Bucket_Boruvka::is_empty(deterministic_bucket_ref())) {
    return {0, ZERO};
  }
  for (size_t i = capacity(); i > 0; --i) {
    if (Bucket_Boruvka::is_good(buckets()[i - 1], seed)) {
      return {buckets()[i - 1].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

void ResizeableAlignedSketchColumn::update(const vec_t update) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  deterministic_bucket_ref() ^= {update, checksum};

  if (depth >= capacity()) {
    reallocate(static_cast<uint8_t>(depth + 1));
  }
  buckets()[depth] ^= {update, checksum};
}

const ColumnEntryDelta ResizeableAlignedSketchColumn::generate_entry_delta(vec_t update) const {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update, seed);
  col_hash_t depth = Bucket_Boruvka::get_index_depth_legacy(update, seed, 60);
  return {Bucket{update, checksum}, static_cast<uint16_t>(depth)};
}

void ResizeableAlignedSketchColumn::apply_entry_delta(const ColumnEntryDelta &delta) {
  deterministic_bucket_ref() ^= delta.bucket;

  if (delta.depth >= capacity()) {
    reallocate(static_cast<uint8_t>(delta.depth + 1));
  }
  buckets()[delta.depth] ^= delta.bucket;
}

void ResizeableAlignedSketchColumn::merge(ResizeableAlignedSketchColumn const& other) {
  deterministic_bucket_ref() ^= other.deterministic_bucket_ref();
  if (other.capacity() > capacity()) {
    reallocate(other.capacity());
  }
  uint32_t *for_vector_merge = reinterpret_cast<uint32_t*>(buckets());
  const uint32_t *other_for_vector_merge = reinterpret_cast<const uint32_t*>(other.buckets());
  int num_vectors = other.capacity() * (sizeof(Bucket) / sizeof(uint32_t));
  hwy::HWY_NAMESPACE::simd_xor(for_vector_merge, other_for_vector_merge, num_vectors);
}

uint8_t ResizeableAlignedSketchColumn::get_depth() const {
  for (size_t i = capacity(); i > 0; --i) {
    if (!Bucket_Boruvka::is_empty(buckets()[i - 1])) {
      return i;
    }
  }
  return 0;
}

size_t ResizeableAlignedSketchColumn::space_usage_bytes() const {
  return sizeof(ResizeableAlignedSketchColumn) + ((static_cast<size_t>(capacity()) + 1) * sizeof(Bucket));
}

static_assert(SketchColumnConcept<FixedSizeSketchColumn, vec_t>,
              "FixedSizeSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableSketchColumn, vec_t>,
              "ResizeableSketchColumn does not satisfy SketchColumnConcept");

static_assert(SketchColumnConcept<ResizeableAlignedSketchColumn, vec_t>,
              "ResizeableAlignedSketchColumn does not satisfy SketchColumnConcept");
