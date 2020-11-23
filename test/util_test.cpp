#include <gtest/gtest.h>
#include "../include/util.h"

TEST(UtilTestSuite, TestNonDirectionalNonSEPairingFn) {
  std::pair<ull,ull> exp;
  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < 1000; ++j) {
      if (i==j) continue;
      exp = {std::min(i,j),std::max(i,j)};
      ASSERT_EQ(exp, inv_nondir_non_self_edge_pairing_fn
      (nondirectional_non_self_edge_pairing_fn(i,j)));
    }
  }
}

TEST(UtilTestSuite, TestNonDirectionNonSEPairingFnOverflow) {
  std::cout << std::numeric_limits<ull>::max() << std::endl;
  std::pair<ull,ull> not_overflow {1ull<<31ull, 1ull<<32ull};
  ASSERT_EQ(not_overflow, inv_nondir_non_self_edge_pairing_fn
  (nondirectional_non_self_edge_pairing_fn(1ull<<31ull, 1ull<<32ull)));
  ASSERT_THROW(nondirectional_non_self_edge_pairing_fn(1ull<<32ull,
                                                       1ull<<33ull),
               std::overflow_error);
}
