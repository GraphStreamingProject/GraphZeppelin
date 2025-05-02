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

TEST(UtilTestSuite, TestSimdXor) {
  uint32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint32_t b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint32_t c[8] = {0};

  hwy::HWY_NAMESPACE::simd_xor(a, b, 5);

  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(a[i], c[i]);
  }
}
