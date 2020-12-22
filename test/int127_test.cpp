#include <gtest/gtest.h>
#include "../include/int127.h"

using namespace boost::multiprecision::literals;

const int128_t bigval = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF_cppi128;
const int128_t negbigval = -0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF_cppi128;
const int128_t toobig = 0x80000000000000000000000000000000_cppi128;
const unsigned long ul = 42069;
const long l = 42069;
const long m = -42069;
const int i = 1337;
const int j = -1337;

TEST(Int127TestSuite, TestConstructorsAndConversions) {

  int127 a {bigval};
  int127 nega {negbigval};
  ASSERT_EQ(a.toBoostUInt128(),bigval);
  ASSERT_EQ(nega.toBoostInt128(),negbigval);
  ASSERT_THROW(nega.toBoostUInt128(), InvalidUInt128CastException);
  ASSERT_THROW(int127 b(toobig), Int127OverflowException);
  int127 iul{ul}, il{l}, im{m}, ii{i}, ij{j};
  ASSERT_EQ(iul.toBoostUInt128(), ul);
  ASSERT_EQ(il.toBoostInt128(),l);
  ASSERT_EQ(im.toBoostInt128(),m);
  ASSERT_EQ(ii.toBoostInt128(),i);
  ASSERT_EQ(ij.toBoostInt128(),j);
}

TEST(Int127TestSuite, TestAssignmentOperators) {
  int127 a = bigval;
  int127 b = negbigval;
  int127 iul = ul, il = l, im = m, ii = i, ij = j;
  ASSERT_EQ(a.toBoostInt128(),bigval);
  ASSERT_EQ(b.toBoostInt128(),negbigval);
  ASSERT_EQ(iul.toBoostInt128(),ul);
  ASSERT_EQ(il.toBoostInt128(),l);
  ASSERT_EQ(im.toBoostInt128(),m);
  ASSERT_EQ(ii.toBoostInt128(),i);
  ASSERT_EQ(ij.toBoostInt128(),j);
}

TEST(Int127TestSuite, TestArithmeticOperators) {
  int127 iul = ul, il = l, im = m, ii = i, ij = j;
  ASSERT_EQ((il+im).toBoostInt128(),l+m);
  ASSERT_EQ((ii-il).toBoostInt128(),i-l);
  ASSERT_EQ((il/ij).toBoostInt128(),l/j);
  ASSERT_EQ((iul%ii).toBoostInt128(), iul%i);

  im += il;
  ASSERT_EQ(im.toBoostInt128(), 0);

  ASSERT_EQ((iul*j).toBoostInt128(), (long) ul*j);
}

TEST(Int127TestSuite, TestRelationalOperators) {
  int127 a {bigval};
  int127 b = bigval;
  int127 c = negbigval;
  ASSERT_EQ(a,b);
  ASSERT_NE(a,c);
  int127 iul = ul, il = l, im = m, ii = i, ij = j;
  ASSERT_GT(il,im);
  ASSERT_GT(il,ii);
  ASSERT_LE(a,b);
  ASSERT_LE(ii,il);
  ASSERT_LE(ij,iul);
  ASSERT_NE(iul, 17ll);
}