#include <gtest/gtest.h>
#include "../include/graph.h"

TEST(GraphTestSuite, DISABLED_SmallGraphConnectivity) {
  unsigned long long int num_nodes = 1000;
  Graph g{num_nodes};
  for (unsigned i=1;i<num_nodes;++i) {
    for (unsigned j = i*2;j<num_nodes;j+=i) {
      g.update({{i,j}, INSERT});
    }
  }
  ASSERT_EQ(2, g.connected_components().size());
}

TEST(GraphTestSuite, DISABLED_IFconnectedComponentsAlgRunTHENupdateLocked) {
  unsigned long long int num_nodes = 1000;
  Graph g{num_nodes};
  for (unsigned i=1;i<num_nodes;++i) {
    for (unsigned j = i*2;j<num_nodes;j+=i) {
      g.update({{i,j}, INSERT});
    }
  }
  g.connected_components();
  ASSERT_THROW(g.update({{1,2}, INSERT}), UpdateLockedException);
  ASSERT_THROW(g.update({{1,2}, DELETE}), UpdateLockedException);
}
