#include "../../include/l0_sampling/sketch.h"
#include <cassert>
#include <cstring>
#include <iostream>

vec_t Sketch::failure_factor = 100;
vec_t Sketch::n;
size_t Sketch::num_elems;
size_t Sketch::num_columns;
size_t Sketch::num_guesses;

/*
 * Static functions for creating sketches with a provided memory location.
 * We use these in the production system to keep supernodes virtually contiguous.
 */
Sketch* Sketch::makeSketch(void* loc, uint64_t seed) {
  return new (loc) Sketch(seed);
}

Sketch* Sketch::makeSketch(void* loc, uint64_t seed, std::istream &binary_in, bool sparse) {
  return new (loc) Sketch(seed, binary_in, sparse);
}

Sketch* Sketch::makeSketch(void* loc, const Sketch& s) {
  return new (loc) Sketch(s);
}

Sketch::Sketch(uint64_t seed): seed(seed) {
  // establish the bucket_a and bucket_c locations
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  // initialize bucket values
  for (size_t i = 0; i < num_elems; ++i) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }
}

Sketch::Sketch(uint64_t seed, std::istream &binary_in, bool sparse): seed(seed) {
  // establish the bucket_a and bucket_c locations
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  if (!sparse) {
    binary_in.read((char*)bucket_a, num_elems * sizeof(vec_t));
    binary_in.read((char*)bucket_c, num_elems * sizeof(vec_hash_t));
  } else {
    for (size_t i = 0; i < num_elems; ++i) {
      bucket_a[i] = 0;
      bucket_c[i] = 0;
    }

    uint16_t idx;
    binary_in.read((char*)&idx, sizeof(idx));
    while (idx < num_elems - 1) {
      binary_in.read((char*)&bucket_a[idx], sizeof(bucket_a[idx]));
      binary_in.read((char*)&bucket_c[idx], sizeof(bucket_c[idx]));
      binary_in.read((char*)&idx, sizeof(idx));
    }
    // finally handle the level 0 bucket (num_elems - 1)
    binary_in.read((char*)&bucket_a[idx], sizeof(bucket_a[idx]));
    binary_in.read((char*)&bucket_c[idx], sizeof(bucket_c[idx]));
  }
}

Sketch::Sketch(const Sketch& s) : seed(s.seed) {
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  std::memcpy(bucket_a, s.bucket_a, num_elems * sizeof(vec_t));
  std::memcpy(bucket_c, s.bucket_c, num_elems * sizeof(vec_hash_t));
}

void Sketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());
  
  // Update depth 0 bucket
  Bucket_Boruvka::update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), num_guesses);
    size_t bucket_id = i * num_guesses + depth;
    likely_if(depth < num_guesses)
      Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, checksum);
  }
}

void Sketch::batch_update(const std::vector<vec_t>& updates) {
  for (const auto& update_idx : updates) {
    update(update_idx);
  }
}

std::pair<vec_t, SampleSketchRet> Sketch::query() {
  if (already_queried) {
    throw MultipleQueryException();
  }
  already_queried = true;

  if (bucket_a[num_elems - 1] == 0 && bucket_c[num_elems - 1] == 0)
    return {0, ZERO}; // the "first" bucket is deterministic so if all zero then no edges to return

  if (Bucket_Boruvka::is_good(bucket_a[num_elems - 1], bucket_c[num_elems - 1], checksum_seed()))
    return {bucket_a[num_elems - 1], GOOD};

  for (unsigned i = 0; i < num_columns; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], checksum_seed()))
        return {bucket_a[bucket_id], GOOD};
    }
  }
  return {0, FAIL};
}

std::pair<std::vector<vec_t>, SampleSketchRet> Sketch::exhaustive_query() {
  unlikely_if (already_queried)
    throw MultipleQueryException();
  std::vector<vec_t> ret;

  unlikely_if (bucket_a[num_elems - 1] == 0 && bucket_c[num_elems - 1] == 0)
    return {ret, ZERO}; // the "first" bucket is deterministic so if zero then no edges to return

  unlikely_if (
  Bucket_Boruvka::is_good(bucket_a[num_elems - 1], bucket_c[num_elems - 1], checksum_seed())) {
    ret.push_back(bucket_a[num_elems - 1]);
    return {ret, GOOD};
  }
  for (unsigned i = 0; i < num_columns; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      unlikely_if (
      Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], checksum_seed())) {
        ret.push_back(bucket_a[bucket_id]);
        update(bucket_a[bucket_id]);
      }
    }
  }
  already_queried = true;

  unlikely_if (ret.size() == 0)
    return {ret, FAIL};
  return {ret, GOOD};
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.seed == sketch2.seed);
  sketch1.already_queried = sketch1.already_queried || sketch2.already_queried;
  if (sketch2.bucket_a[Sketch::num_elems-1] == 0 && sketch2.bucket_c[Sketch::num_elems-1] == 0)
    return sketch1;
  for (unsigned i = 0; i < Sketch::num_elems; i++) {
    sketch1.bucket_a[i] ^= sketch2.bucket_a[i];
    sketch1.bucket_c[i] ^= sketch2.bucket_c[i];
  }
  return sketch1;
}

bool operator== (const Sketch &sketch1, const Sketch &sketch2) {
  if (sketch1.seed != sketch2.seed || sketch1.already_queried != sketch2.already_queried) 
    return false;

  for (size_t i = 0; i < Sketch::num_elems; ++i) {
    if (sketch1.bucket_a[i] != sketch2.bucket_a[i]) return false;
  }

  for (size_t i = 0; i < Sketch::num_elems; ++i) {
    if (sketch1.bucket_c[i] != sketch2.bucket_c[i]) return false;
  }

  return true;
}

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  vec_t a      = sketch.bucket_a[Sketch::num_elems - 1];
  vec_hash_t c = sketch.bucket_c[Sketch::num_elems - 1];
  bool good    = Bucket_Boruvka::is_good(a, c, sketch.checksum_seed());

  os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;

  for (unsigned i = 0; i < Sketch::num_columns; ++i) {
    for (unsigned j = 0; j < Sketch::num_guesses; ++j) {
      unsigned bucket_id = i * Sketch::num_guesses + j;
      vec_t a      = sketch.bucket_a[bucket_id];
      vec_hash_t c = sketch.bucket_c[bucket_id];
      bool good    = Bucket_Boruvka::is_good(a, c, sketch.checksum_seed());

      os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;
    }
    os << std::endl;
  }
  return os;
}

void Sketch::write_binary(std::ostream& binary_out) {
  const_cast<const Sketch*>(this)->write_binary(binary_out);
}

void Sketch::write_binary(std::ostream& binary_out) const {
  // Write out the bucket values to the stream.
  binary_out.write((char*)bucket_a, num_elems * sizeof(vec_t));
  binary_out.write((char*)bucket_c, num_elems * sizeof(vec_hash_t));
}

void Sketch::write_sparse_binary(std::ostream& binary_out) {
  const_cast<const Sketch*>(this)->write_sparse_binary(binary_out);
}

void Sketch::write_sparse_binary(std::ostream& binary_out) const {
  for (uint16_t i = 0; i < num_elems - 1; i++) {
    if (bucket_a[i] == 0 && bucket_c[i] == 0)
      continue;
    binary_out.write((char*)&i, sizeof(i));
    binary_out.write((char*)&bucket_a[i], sizeof(vec_t));
    binary_out.write((char*)&bucket_c[i], sizeof(vec_hash_t));
  }
  // Always write down the deterministic bucket to mark the end of the Sketch
  uint16_t index = num_elems - 1;
  binary_out.write((char*)&index, sizeof(index));
  binary_out.write((char*)&bucket_a[num_elems-1], sizeof(vec_t));
  binary_out.write((char*)&bucket_c[num_elems-1], sizeof(vec_hash_t));
}
