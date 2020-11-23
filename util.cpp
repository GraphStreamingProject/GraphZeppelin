#include <stdexcept>
#include <boost/multiprecision/cpp_int.hpp>
#include "include/util.h"

using uint128_t = boost::multiprecision::uint128_t;

const ull ULLMAX = std::numeric_limits<ull>::max();

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

std::pair<ull, ull> inv_nondir_non_self_edge_pairing_fn(ull idx) {
  uint128_t eidx = 8*(uint128_t)idx + 1;
  eidx = sqrt(eidx)+1;
  eidx/=2;
  ull i,j = (ull) eidx;
  if (j%2 == 0) i = idx-(j>>1u)*(j-1);
  else i = idx-j*((j-1)>>1u);
  return {i, j};
}

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
