#pragma once
#include <cmath>

inline static double binomcdf(unsigned long x, unsigned long n, double p) {
  long double total = 0;
  long double term = std::pow(static_cast<long double>(1 - p), n);
  for (unsigned long i = 0; i <= x; i++) {
    total += term;
    term *= (n - i) * p / (i + 1) / (1 - p);
  }
  return total;
}
