#pragma once
#include <format>
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
  { t.update(std::declval<V>()) } -> std::same_as<void>;
  { t.merge(other) } -> std::same_as<void>;

  { t.clear()} -> std::same_as<void>;
  { t.zero_contents()} -> std::same_as<void>;

  { t.get_depth() } -> std::same_as<uint8_t>;
  { t.get_seed() } -> std::same_as<uint64_t>;
  
  { t.serialize(std::declval<std::ostream&>()) };
  { t.reset_sample_state()} -> std::same_as<void>;
  { t == other } -> std::same_as<bool>;
  // copy constructor required
  requires std::constructible_from<T, const T&>;
  // constructor with capacity hint, column index for seeding
  requires std::constructible_from<T, uint8_t, uint16_t>;
  { T::suggest_capacity(std::declval<size_t>()) } -> std::same_as<uint8_t>;
};

/*
  TODOs - 
  1) Define a vertex group level sketch concept
*/
