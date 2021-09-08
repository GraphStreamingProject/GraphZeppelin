#include "../include/sketch.h"
#include <cassert>
#include <iostream>

double Sketch::num_bucket_factor = 0.5;
vec_t Sketch::n;
size_t Sketch::num_elems;

Sketch::SketchUniquePtr Sketch::makeSketch(const Sketch &old) {
  return makeSketch(old.seed);
}

Sketch::SketchUniquePtr Sketch::makeSketch(long seed) {
  void* loc = malloc(sketchSizeof());
  return SketchUniquePtr(makeSketch(loc, seed), [](Sketch* s){ s->~Sketch(); free(s); });
}

Sketch::SketchUniquePtr Sketch::makeSketch(long seed, std::fstream &binary_in) {
  void* loc = malloc(sketchSizeof());
  return SketchUniquePtr(makeSketch(loc, seed, binary_in), [](Sketch* s){ free(s); });
}

Sketch* Sketch::makeSketch(void* loc, long seed, std::fstream &binary_in) {
  return new (loc) Sketch(seed, binary_in);
}

Sketch* Sketch::makeSketch(void* loc, long seed) {
  return new (loc) Sketch(seed);
}

Sketch::Sketch(long seed): seed(seed) {
  // establish the bucket_a and bucket_c locations and number of elements
  num_elems = get_num_elems();
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));


  for (size_t i = 0; i < num_elems; ++i) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }
}

Sketch::Sketch(long seed, std::fstream &binary_in): seed(seed) {
  // establish the bucket_a and bucket_c locations and number of elements
  num_elems = get_num_elems();
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  binary_in.read((char*)bucket_a, num_elems * sizeof(vec_t));
  binary_in.read((char*)bucket_c, num_elems * sizeof(vec_hash_t));
}

void Sketch::update(const vec_t& update_idx) {
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  XXH64_hash_t update_hash = Bucket_Boruvka::index_hash(update_idx, seed);
  for (unsigned i = 0; i < num_buckets; ++i) {
    col_hash_t col_index_hash = Bucket_Boruvka::col_index_hash(i, update_idx, seed);
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::contains(col_index_hash, 1 << j)) {
        Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, update_hash);
      } else break;
    }
  }
}

void Sketch::batch_update(const std::vector<vec_t>& updates) {
  for (const auto& update_idx : updates) {
    update(update_idx);
  }
}

vec_t Sketch::query() {
  if (already_quered) {
    throw MultipleQueryException();
  }
  already_quered = true;
  bool all_buckets_zero = true;
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (bucket_a[bucket_id] != 0 || bucket_c[bucket_id] != 0) {
        all_buckets_zero = false;
      }
      if (Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], n, i, 1 << j, seed)) {
        return bucket_a[bucket_id];
      }
    }
  }
  if (all_buckets_zero) {
    throw AllBucketsZeroException();
  } else {
    throw NoGoodBucketException();
  }
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.seed == sketch2.seed);
  for (unsigned i = 0; i < Sketch::num_elems; i++){
    sketch1.bucket_a[i] ^= sketch2.bucket_a[i];
    sketch1.bucket_c[i] ^= sketch2.bucket_c[i];
  }
  sketch1.already_quered = sketch1.already_quered || sketch2.already_quered;
  return sketch1;
}

bool operator== (const Sketch &sketch1, const Sketch &sketch2) {
  if (sketch1.seed != sketch2.seed || sketch1.already_quered != sketch2.already_quered) 
    return false;

  for (size_t i = 0; i < Sketch::num_elems; ++i)
  {
    if (sketch1.bucket_a[i] != sketch2.bucket_a[i]) return false;
  }

  for (size_t i = 0; i < Sketch::num_elems; ++i)
  {
    if (sketch1.bucket_c[i] != sketch2.bucket_c[i]) return false;
  }

  return true;
}

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  const unsigned long long int num_buckets = bucket_gen(sketch.n, sketch.num_bucket_factor);
  const unsigned long long int num_guesses = guess_gen(sketch.n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      for (unsigned k = 0; k < sketch.n; k++) {
        os << (Bucket_Boruvka::contains(Bucket_Boruvka::col_index_hash(i, k, sketch.seed), 1 << j) ? '1' : '0');
      }
      os << std::endl
         << "a:" << sketch.bucket_a[bucket_id] << std::endl
         << "c:" << sketch.bucket_c[bucket_id] << std::endl
         << (Bucket_Boruvka::is_good(sketch.bucket_a[bucket_id], sketch.bucket_c[bucket_id], sketch.n, i, 1 << j, sketch.seed) ? "good" : "bad") << std::endl;
    }
  }
  return os;
}

void Sketch::write_binary(std::fstream& binary_out) {
  binary_out.write((char*)bucket_a, num_elems * sizeof(vec_t));
  binary_out.write((char*)bucket_c, num_elems * sizeof(vec_hash_t));
}
