#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "include/supernode.h"

typedef long long int ll;
typedef unsigned long long int ull;

using uint128_t = boost::multiprecision::uint128_t;

const ull ULLMAX = std::numeric_limits<ull>::max();

/**
 * A function Z_+ x Z_+ -> Z_+ that implements a non-self-edge pairing function
 * that does not respect order of inputs.
 * That is, (2,2) would not be a valid input. (1,3) and (3,1) would be treated as
 * identical inputs.
 * @return i + j*(j-1)/2
 * @throws overflow_error if there would be an overflow in computing the function.
 */
ull nondirectional_non_self_edge_pairing_fn(ull i, ull j) {
  // swap i,j if necessary
  if (i > j) {
    i^=j;
    j^=i;
    i^=j;
  }

  ull jm = j-1;
  if (j%2 == 0) j>>=1u;
  else jm>>=1u;
  if (ULLMAX/j < jm)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  j*=jm;
  if (ULLMAX-j < i)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  return i+j;
}

pair<ull, ull> inv_nondir_non_self_edge_pairing_fn(ull idx) {
  uint128_t eidx = 8*(uint128_t)idx + 1;
  eidx = sqrt(eidx)+1;
  eidx/=2;
  ull i,j = (ull) eidx;
  if (j%2 == 0) i = idx-(j>>1u)*(j-1);
  else i = idx-j*((j-1)>>1u);
  return {i, j};
}

/**
 * Implementation of the Cantor diagonal pairing function.
 * @throws overflow_error if there would be an overflow in computing the function.
 */
ull cantor_pairing_fn(ull i, ull j) {
  if (ULLMAX - i < j)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  if (j < ULLMAX && ULLMAX - i < j+1)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  ull am = i+j, bm = i+j+1;
  if (am % 2 == 0) am>>=1u;
  else bm>>=1u;
  if (ULLMAX/am < bm)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  am*=bm;
  if (ULLMAX - am < j)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  return am+j;
}

Supernode::Supernode(ull n, long seed): sketches(log2(n)), idx(0), logn(log2
(n)) {
  // generate logn sketches for each supernode (read: node)
  srand(seed);
  long r = rand();
  for (int i=0;i<logn;++i)
    sketches[i] = new Sketch(n*n, r);
}

boost::optional<Edge> Supernode::sample() {
  if (idx == logn) throw OutOfQueriesException();
  Update query;
  try {
    query = sketches[idx]->query();
  } catch (AllBucketsZeroException &e) {
    ++idx;
    return {};
  }
  ++idx;
  if (query.delta == 0) return {};
  return inv_nondir_non_self_edge_pairing_fn(query.index);
}

void Supernode::merge(Supernode &other) {
  idx = max(idx, other.idx);
  for (int i=idx;i<logn;++i) {
    (*sketches[i])+=(*other.sketches[i]);
  }
}

void Supernode::update(pair<Edge, int> update) {
  Update upd = {nondirectional_non_self_edge_pairing_fn(update.first.first,
        update.first.second), update.second};
  for (Sketch* s : sketches)
    s->update(upd);
}
