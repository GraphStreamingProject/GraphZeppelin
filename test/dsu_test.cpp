
#include <gtest/gtest.h>
#include "types.h"
#include "dsu.h"

// must be power of 2
constexpr size_t slots_power = 16;
constexpr size_t num_slots = 1 << slots_power;

TEST(DSU_Tests, SerialDSU) {
  DisjointSetUnion<node_id_t> dsu(num_slots);

  // merge fully
  for (node_id_t p = slots_power; p > 0; p--) {
    size_t active = 1 << p;
    for (node_id_t i = 0; i < active / 2; i++) {
      dsu.merge(i, i + active / 2);
    }
  }

  node_id_t root = dsu.find_root(0);
  for (node_id_t i = 1; i < num_slots; i++) {
    ASSERT_EQ(root, dsu.find_root(i));
  }
}

TEST(DSU_Tests, DSU_MT_Single_Thread) {
  DisjointSetUnion_MT<node_id_t> dsu(num_slots);

  // merge fully
  for (node_id_t p = slots_power; p > 0; p--) {
    size_t active = 1 << p;
    for (node_id_t i = 0; i < active / 2; i++) {
      dsu.merge(i, i + active / 2);
    }
  }

  node_id_t root = dsu.find_root(0);
  for (node_id_t i = 1; i < num_slots; i++) {
    ASSERT_EQ(root, dsu.find_root(i));
  }
}

TEST(DSU_Tests, DSU_MT_Eight_Threads) {
  DisjointSetUnion_MT<node_id_t> dsu(num_slots);

  // merge fully
#pragma omp parallel for num_threads(8)
  for (node_id_t p = slots_power; p > 0; p--) {
    size_t active = 1 << p;
    for (node_id_t i = 0; i < active / 2; i++) {
      dsu.merge(i, i + active / 2);
    }
  }

  node_id_t root = dsu.find_root(0);
  for (node_id_t i = 1; i < num_slots; i++) {
    ASSERT_EQ(root, dsu.find_root(i));
  }
}

