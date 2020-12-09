#include <stdexcept>
#include <boost/multiprecision/cpp_int.hpp>
#include "include/util.h"

using uint128_t = boost::multiprecision::uint128_t;

typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ull>::max();

unsigned long long int double_to_ull(double d, double epsilon) {
  return (unsigned long long) (d + epsilon);
}

ull nondirectional_non_self_edge_pairing_fn(ull i, ull j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  ull jm = j-1ull;
  if ((j & 1ull) == 0ull) j>>=1ull;
  else jm>>=1ull;
  if (ULLMAX/j < jm)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  j*=jm;
  if (ULLMAX-j < i)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  return i+j;
}

std::pair<ull, ull> inv_nondir_non_self_edge_pairing_fn(ull idx) {
  uint128_t eidx = 8ull*(uint128_t)idx + 1ull;
  eidx = sqrt(eidx)+1ull;
  eidx/=2ull;
  ull i,j = (ull) eidx;
  if ((j & 1ull) == 0ull) i = idx-(j>>1ull)*(j-1ull);
  else i = idx-j*((j-1ull)>>1ull);
  return {i, j};
}

ull cantor_pairing_fn(ull i, ull j) {
  if (ULLMAX - i < j)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  if (j < ULLMAX && ULLMAX - i < j+1ull)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  ull am = i+j, bm = i+j+1ull;
  if ((am & 1ull) == 0ull) am>>=1ull;
  else bm>>=1ull;
  if (ULLMAX/am < bm)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  am*=bm;
  if (ULLMAX - am < j)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  return am+j;
}
