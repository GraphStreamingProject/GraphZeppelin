
#include <gtest/gtest.h>
#include "types.h"
#include "dsu.h"

// must be power of 2
constexpr size_t num_slots = 1024;

TEST(DSU_Tests, SerialDSU) {
  DisjointSetUnion<node_id_t> dsu(num_slots);

  // merge fully
  size_t active = num_slots;
  while (active > 1) {
    for (node_id_t i = 0; i < active / 2; i++) {
      dsu.merge(i, i + active / 2);
    }
    for (node_id_t i = 0; i < active / 2; i++) {
      ASSERT_EQ(dsu.find_root(i), dsu.find_root(i + active / 2));
    }
    active /= 2;
  }

  node_id_t root = dsu.find_root(0);
  for (node_id_t i = 1; i < num_slots; i++) {
    ASSERT_EQ(root, dsu.find_root(i));
  }
}

TEST(DSU_Tests, DSU_MT_Single_Thread) {
  DisjointSetUnion_MT<node_id_t> dsu(num_slots);

  // merge fully
  size_t active = num_slots;
  while (active > 1) {
    for (node_id_t i = 0; i < active / 2; i++) {
      dsu.merge(i, i + active / 2);
    }
    for (node_id_t i = 0; i < active / 2; i++) {
      ASSERT_EQ(dsu.find_root(i), dsu.find_root(i + active / 2));
    }
    active /= 2;
  }

  node_id_t root = dsu.find_root(0);
  for (node_id_t i = 1; i < num_slots; i++) {
    ASSERT_EQ(root, dsu.find_root(i));
  }
}

TEST(DSU_Tests, DSU_MT_Eight_Threads) {
  DisjointSetUnion_MT<node_id_t> dsu(num_slots);

  // merge fully
  size_t active = num_slots;
  while (active > 1) {
#pragma omp parallel for num_threads(8)
    for (node_id_t i = 0; i < active / 2; i++) {
      dsu.merge(i, i + active / 2);
    }
    for (node_id_t i = 0; i < active / 2; i++) {
      ASSERT_EQ(dsu.find_root(i), dsu.find_root(i + active / 2));
    }
    active /= 2;
  }

  node_id_t root = dsu.find_root(0);
  for (node_id_t i = 1; i < num_slots; i++) {
    ASSERT_EQ(root, dsu.find_root(i));
  }
}

