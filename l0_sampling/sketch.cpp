#include "../include/sketch.h"
#include <cassert>

Sketch::Sketch(vec_t n, long seed, double num_bucket_factor):
    seed(seed), n(n), num_bucket_factor(num_bucket_factor) {
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  bucket_a = std::vector<vec_t>(num_buckets * num_guesses);
  bucket_c = std::vector<vec_hash_t>(num_buckets * num_guesses);
}

Sketch::Sketch(const Sketch &old) : seed(old.seed), n(old.n),
                                    num_bucket_factor(old.num_bucket_factor), already_quered(old.already_quered) {
  bucket_a = old.bucket_a;
  bucket_c = old.bucket_c;
}

Sketch::Sketch() : seed(0), n(0), num_bucket_factor(1.0), bucket_a(), bucket_c(){}

void Sketch::update(const vec_t& update_idx) {
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  XXH64_hash_t update_hash = Bucket_Boruvka::index_hash(update_idx, seed);
  for (unsigned i = 0; i < num_buckets; ++i) {
    col_hash_t col_index_hash = Bucket_Boruvka::col_index_hash(i, update_idx, seed);
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::contains(col_index_hash, 1 << j)){
        Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, update_hash);
      } else break;
    }
  }
}

void Sketch::clear()
{
  bucket_a = std::vector<vec_t>(bucket_a.size());
  bucket_c = std::vector<vec_hash_t>(bucket_c.size());
  already_quered = false;
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

Sketch &Sketch::operator=(const Sketch& other)
{
  const_cast<long&>(seed) = other.seed;
  const_cast<vec_t&>(n) = other.n;
  const_cast<double&>(num_bucket_factor) = other.num_bucket_factor;
  bucket_a = other.bucket_a;
  bucket_c = other.bucket_c;
  already_quered = other.already_quered;
  return *this;
}


Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.n == sketch2.n);
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.num_bucket_factor == sketch2.num_bucket_factor);
  for (unsigned i = 0; i < sketch1.bucket_a.size(); i++){
    sketch1.bucket_a[i] ^= sketch2.bucket_a[i];
    sketch1.bucket_c[i] ^= sketch2.bucket_c[i];
  }
  sketch1.already_quered = sketch1.already_quered || sketch2.already_quered;
  return sketch1;
}

bool operator== (const Sketch &sketch1, const Sketch &sketch2) {
  return sketch1.n == sketch2.n && sketch1.seed == sketch2.seed &&
    sketch1.num_bucket_factor == sketch2.num_bucket_factor &&
    sketch1.bucket_a == sketch2.bucket_a &&
    sketch1.bucket_c == sketch2.bucket_c &&
    sketch1.already_quered == sketch2.already_quered;
}

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  const unsigned long long int num_buckets = bucket_gen(sketch.n, sketch.num_bucket_factor);
  const unsigned long long int num_guesses = guess_gen(sketch.n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      //for (unsigned k = 0; k < sketch.n; k++) {
      //  os << (Bucket_Boruvka::contains(Bucket_Boruvka::col_index_hash(i, k, sketch.seed), 1 << j) ? '1' : '0');
      //}
      os << "bucket id:" << bucket_id << std::endl
         << "a:" << sketch.bucket_a[bucket_id] << std::endl
         << "c:" << sketch.bucket_c[bucket_id] << std::endl
         << (Bucket_Boruvka::is_good(sketch.bucket_a[bucket_id], sketch.bucket_c[bucket_id], sketch.n, i, 1 << j, sketch.seed) ? "good" : "bad") << std::endl;
    }
  }
  return os;
}
