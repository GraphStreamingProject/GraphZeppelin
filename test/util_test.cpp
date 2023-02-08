#include <gtest/gtest.h>
#include "../include/util.h"

TEST(UtilTestSuite, TestConcatPairingFn) {
  Edge exp;
  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < 1000; ++j) {
      if (i==j) continue;
      exp.src = std::min(i,j);
      exp.dst = std::max(i,j);
      ASSERT_EQ(exp.src, inv_concat_pairing_fn(concat_pairing_fn(i, j)).src);
      ASSERT_EQ(exp.dst, inv_concat_pairing_fn(concat_pairing_fn(i, j)).dst);
    }
  }
}
