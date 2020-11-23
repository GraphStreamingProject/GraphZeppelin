#include <gtest/gtest.h>
#include "../include/util.h"

TEST(UtilTestSuite, TestNonDirectionalNonSEPairingFn) {
  std::pair<ull,ull> exp;
  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < 1000; ++j) {
      exp = {i,j};
      ASSERT_EQ(exp, inv_nondir_non_self_edge_pairing_fn
      (nondirectional_non_self_edge_pairing_fn(i,j)));
    }
  }
}

TEST(UtilTestSuite, TestNonDirectionNonSEPairingFnOverflow) {

}