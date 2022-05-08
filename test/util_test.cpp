#include <gtest/gtest.h>
#include "../include/util.h"

TEST(UtilTestSuite, TestConcatPairingFn) {
  std::pair<uint32_t,uint32_t> exp;
  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < 1000; ++j) {
      if (i==j) continue;
      exp = {std::min(i,j),std::max(i,j)};
      ASSERT_EQ(exp, inv_concat_pairing_fn
            (concat_pairing_fn(i, j)));
    }
  }
}
