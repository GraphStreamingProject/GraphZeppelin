#pragma once
#include <concepts>
#include "bucket.h"
#include <unordered_set>

enum SampleResult {
  GOOD,  // sampling this sketch returned a single non-zero value
  ZERO,  // sampling this sketch returned that there are no non-zero values
  FAIL   // sampling this sketch failed to produce a single non-zero value
};

template <typename T = vec_t> requires(std::integral<T>)
struct SketchSample {
  T idx;
  SampleResult result;
};

// TODO - figure out how to template this instead of using vec_t
struct ColumnEntryDelta {
  Bucket bucket;
  uint16_t depth;
};


template <typename T = vec_t> requires(std::integral<T>)
struct ExhaustiveSketchSample {
  std::unordered_set<T> idxs;
  SampleResult result;
};

template <typename T, typename V>
concept ConnectivitySketchConcept = requires(T t, T other) {
  { t.sample() } -> std::same_as<SketchSample<V>>;
  { t.clear()} -> std::same_as<void>;
  { t.update(std::declval<V>()) };
  { t.merge(std::declval<T>()) };
  { t.range_merge(std::declval<T>(), std::declval<size_t>(), std::declval<size_t>()) };
  { t.serialize(std::declval<std::ostream&>()) };
  { t == other } -> std::same_as<bool>;
  requires std::constructible_from<T, const T&>;
};

template <typename T, typename V>
concept SketchColumnConcept = requires(T t, T other) {
  { t.sample() } -> std::same_as<SketchSample<V>>;
  { t.generate_entry_delta(std::declval<V>()) } -> std::same_as<const ColumnEntryDelta>;
  { t.apply_entry_delta(std::declval<const ColumnEntryDelta>()) } -> std::same_as<void>;
  { t.update(std::declval<V>()) } -> std::same_as<void>;
  // require an atomic_update function. 
  // up to implementer whether it uses locks or simply atomic XOR 
  // (note that only the prior works for fixed size sketches)
  { t.atomic_update(std::declval<V>()) } -> std::same_as<void>;
  
  { t.merge(other) } -> std::same_as<void>;
  
  { t.is_initialized() } -> std::same_as<bool>;

  { t.clear()} -> std::same_as<void>;
  { t.zero_contents()} -> std::same_as<void>;

  { t.get_depth() } -> std::same_as<uint8_t>;
  { t.get_seed() } -> std::same_as<uint64_t>;
  
  { t.serialize(std::declval<std::ostream&>()) };
  { t.reset_sample_state()} -> std::same_as<void>;
  { t == other } -> std::same_as<bool>;
  // copy constructor required
  // requires std::constructible_from<T, const T&>;
  requires std::copy_constructible<T>;
  // move constructor and assignment required
  requires std::move_constructible<T>;
  requires std::assignable_from<T&, T>;
  // constructor with capacity hint, and a seed
  requires std::constructible_from<T, uint8_t, uint64_t>;
  { T::suggest_capacity(std::declval<size_t>()) } -> std::same_as<uint8_t>;
};

/*
  TODOs - 
  1) Define a vertex group level sketch concept
*/
