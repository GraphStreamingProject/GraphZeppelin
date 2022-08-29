#include <gtest/gtest.h>
#include <fstream>
#include <algorithm>
#include "../include/graph.h"
#include "../graph_worker.h"
#include "../include/test/file_graph_verifier.h"
#include "../include/test/mat_graph_verifier.h"
#include "../include/test/graph_gen.h"
#include <binary_graph_stream.h>

/**
 * For many of these tests (especially for those upon very sparse and small graphs)
 * we allow for a certain number of failures per test.
 * This is because the responsibility of these tests is to quickly alert us 
 * to “this code is very wrong” whereas the statistical testing is responsible 
 * for a more fine grained analysis.
 * In this context a false positive is much worse than a false negative.
 * With 2 failures allowed per test our entire testing suite should fail 1/5000 runs.
 */

// We create this class and instantiate a paramaterized test suite so that we
// can run these tests both with the GutterTree and with StandAloneGutters
class GraphTest : public testing::TestWithParam<GutterSystem> {

};
INSTANTIATE_TEST_SUITE_P(GraphTestSuite, GraphTest, testing::Values(GUTTERTREE, STANDALONE, CACHETREE));

TEST_P(GraphTest, SmallGraphConnectivity) {
  auto config = GraphConfiguration().gutter_sys(GetParam());
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  node_id_t num_nodes;
  in >> num_nodes;
  edge_id_t m;
  in >> m;
  node_id_t a, b;
  Graph g{num_nodes, config};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
  ASSERT_EQ(78, g.connected_components().size());
}

TEST(GraphTest, IFconnectedComponentsAlgRunTHENupdateLocked) {
  auto config = GraphConfiguration().gutter_sys(STANDALONE);
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  node_id_t num_nodes;
  in >> num_nodes;
  edge_id_t m;
  in >> m;
  node_id_t a, b;
  Graph g{num_nodes, config};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
  g.connected_components();
  ASSERT_THROW(g.update({{1,2}, INSERT}), UpdateLockedException);
  ASSERT_THROW(g.update({{1,2}, DELETE}), UpdateLockedException);
}

TEST(GraphTest, TestSupernodeRestoreAfterCCFailure) {
  for (int s = 0; s < 2; s++) {
    auto config = GraphConfiguration().backup_in_mem(s == 0);
    const std::string fname = __FILE__;
    size_t pos = fname.find_last_of("\\/");
    const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
    std::ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
    node_id_t num_nodes;
    in >> num_nodes;
    edge_id_t m;
    in >> m;
    node_id_t a, b;
    Graph g{num_nodes, config};
    while (m--) {
      in >> a >> b;
      g.update({{a, b}, INSERT});
    }
    g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
    g.should_fail_CC();

    // flush to make sure copy supernodes is consistent with graph supernodes
    g.gts->force_flush();
    GraphWorker::pause_workers();
    Supernode* copy_supernodes[num_nodes];
    for (node_id_t i = 0; i < num_nodes; ++i) {
      copy_supernodes[i] = Supernode::makeSupernode(*g.supernodes[i]);
    }

    ASSERT_THROW(g.connected_components(true), OutOfQueriesException);
    for (node_id_t i = 0; i < num_nodes; ++i) {
      for (int j = 0; j < copy_supernodes[i]->get_num_sktch(); ++j) {
        ASSERT_TRUE(*copy_supernodes[i]->get_sketch(j) ==
                  *g.supernodes[i]->get_sketch(j));
      }
    }
  }
}

TEST_P(GraphTest, TestCorrectnessOnSmallRandomGraphs) {
  auto config = GraphConfiguration().gutter_sys(GetParam());
  int num_trials = 5;
  while (num_trials--) {
    generate_stream();
    std::ifstream in{"./sample.txt"};
    node_id_t n;
    edge_id_t m;
    in >> n >> m;
    Graph g{n, config};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }

    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    g.connected_components();
  }
}

TEST_P(GraphTest, TestCorrectnessOnSmallSparseGraphs) {
  auto config = GraphConfiguration().gutter_sys(GetParam());
  int num_trials = 5;
  while(num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
    std::ifstream in{"./sample.txt"};
    node_id_t n;
    edge_id_t m;
    in >> n >> m;
    Graph g{n, config};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }

    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    g.connected_components();
  } 
}

TEST_P(GraphTest, TestCorrectnessOfReheating) {
  auto config = GraphConfiguration().gutter_sys(GetParam());
  int num_trials = 5;
  while (num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
    std::ifstream in{"./sample.txt"};
    node_id_t n;
    edge_id_t m;
    in >> n >> m;
    Graph *g = new Graph (n, config);
    int type, a, b;
    printf("number of updates = %lu\n", m);
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) g->update({{a, b}, INSERT});
      else g->update({{a, b}, DELETE});
    }
    g->write_binary("./out_temp.txt");
    g->set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    std::vector<std::set<node_id_t>> g_res;
    g_res = g->connected_components();
    printf("number of CC = %lu\n", g_res.size());
    delete g; // delete g to avoid having multiple graphs open at once. Which is illegal.

    Graph reheated {"./out_temp.txt"};
    reheated.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    auto reheated_res = reheated.connected_components();
    printf("number of reheated CC = %lu\n", reheated_res.size());
    ASSERT_EQ(g_res.size(), reheated_res.size());
    for (unsigned i = 0; i < g_res.size(); ++i) {
      std::vector<node_id_t> symdif;
      std::set_symmetric_difference(g_res[i].begin(), g_res[i].end(),
          reheated_res[i].begin(), reheated_res[i].end(),
          std::back_inserter(symdif));
      ASSERT_EQ(0, symdif.size());
    }
  }
}

// Test the multithreaded system by specifiying multiple
// Graph Workers of size 2. Ingest a stream and run CC algorithm.
TEST_P(GraphTest, MultipleWorkers) {
  auto config = GraphConfiguration()
                .gutter_sys(GetParam())
                .num_groups(4)
                .group_size(2);
  int num_trials = 5;
  while(num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
    std::ifstream in{"./sample.txt"};
    node_id_t n;
    edge_id_t m;
    in >> n >> m;
    Graph g{n, config};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }

    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    g.connected_components();
  } 
}

TEST_P(GraphTest, TestPointQuery) {
  auto config = GraphConfiguration().gutter_sys(GetParam());
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  node_id_t num_nodes;
  in >> num_nodes;
  edge_id_t m;
  in >> m;
  node_id_t a, b;
  Graph g{num_nodes, config};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
  std::vector<std::set<node_id_t>> ret = g.connected_components(true);
  std::vector<node_id_t> ccid (num_nodes);
  for (node_id_t i = 0; i < ret.size(); ++i) {
    for (const node_id_t node : ret[i]) {
      ccid[node] = i;
    }
  }
  for (node_id_t i = 0; i < std::min(10u, num_nodes); ++i) {
    for (node_id_t j = 0; j < std::min(10u, num_nodes); ++j) {
      g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
      ASSERT_EQ(g.point_query(i, j), ccid[i] == ccid[j]);
    }
  }
}

TEST(GraphTest, TestQueryDuringStream) {
  auto config = GraphConfiguration()
                .gutter_sys(STANDALONE)
                .backup_in_mem(false);
  { // test copying to disk
    generate_stream({1024, 0.002, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});
    std::ifstream in{"./sample.txt"};
    node_id_t n;
    edge_id_t m;
    in >> n >> m;
    Graph g(n, config);
    MatGraphVerifier verify(n);

    int type;
    node_id_t a, b;
    edge_id_t tenth = m / 10;
    for(int j = 0; j < 9; j++) {
      for (edge_id_t i = 0; i < tenth; i++) {
        in >> type >> a >> b;
        g.update({{a,b}, (UpdateType)type});
        verify.edge_update(a, b);
      }
      verify.reset_cc_state();
      g.set_verifier(std::make_unique<MatGraphVerifier>(verify));
      g.connected_components(true);
    }
    m -= 9 * tenth;
    while(m--) {
      in >> type >> a >> b;
      g.update({{a,b}, (UpdateType)type});
      verify.edge_update(a, b);
    }
    verify.reset_cc_state();
    g.set_verifier(std::make_unique<MatGraphVerifier>(verify));
    g.connected_components();
  }

  config.backup_in_mem(true);
  { // test copying in memory
    generate_stream({1024, 0.002, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});
    std::ifstream in{"./sample.txt"};
    node_id_t n;
    edge_id_t m;
    in >> n >> m;
    Graph g(n, config);
    MatGraphVerifier verify(n);

    int type;
    node_id_t a, b;
    edge_id_t tenth = m / 10;
    for(int j = 0; j < 9; j++) {
      for (edge_id_t i = 0; i < tenth; i++) {
        in >> type >> a >> b;
        g.update({{a,b}, (UpdateType)type});
        verify.edge_update(a, b);
      }
      verify.reset_cc_state();
      g.set_verifier(std::make_unique<MatGraphVerifier>(verify));
      g.connected_components(true);
    }
    m -= 9 * tenth;
    while(m--) {
      in >> type >> a >> b;
      g.update({{a,b}, (UpdateType)type});
      verify.edge_update(a, b);
    }
    verify.reset_cc_state();
    g.set_verifier(std::make_unique<MatGraphVerifier>(verify));
    g.connected_components();
  }
}

TEST(GraphTest, EagerDSUTest) {
  node_id_t num_nodes = 100;
  Graph g{num_nodes};
  MatGraphVerifier verify(num_nodes);

  // This should be a spanning forest edge
  g.update({{1, 2}, INSERT});
  verify.edge_update(1, 2);
  verify.reset_cc_state();
  g.set_verifier(std::make_unique<decltype(verify)>(verify));
  g.connected_components(true);

  // This should be a spanning forest edge
  g.update({{2, 3}, INSERT});
  verify.edge_update(2, 3);
  verify.reset_cc_state();
  g.set_verifier(std::make_unique<decltype(verify)>(verify));
  g.connected_components(true);

  // This should be an edge within a component
  g.update({{1, 3}, INSERT});
  verify.edge_update(1, 3);
  verify.reset_cc_state();
  g.set_verifier(std::make_unique<decltype(verify)>(verify));
  g.connected_components(true);

  // This should delete an edge within a component
  g.update({{1, 3}, DELETE});
  verify.edge_update(1, 3);
  verify.reset_cc_state();
  g.set_verifier(std::make_unique<decltype(verify)>(verify));
  g.connected_components(true);

  // This one should delete a spanning forest edge and cause a rebuild
  g.update({{2, 3}, DELETE});
  verify.edge_update(2, 3);
  verify.reset_cc_state();
  g.set_verifier(std::make_unique<decltype(verify)>(verify));
  g.connected_components(true);

  // This one should be a normal edge
  g.update({{2, 3}, INSERT});
  verify.edge_update(2, 3);
  verify.reset_cc_state();
  g.set_verifier(std::make_unique<decltype(verify)>(verify));
  g.connected_components(true);
}

TEST(GraphTest, MultipleInsertThreads) {
  auto config = GraphConfiguration().gutter_sys(STANDALONE);
  int num_threads = 4;

  generate_stream({1024, 0.2, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});
  std::ifstream in{"./sample.txt"};
  node_id_t n;
  edge_id_t m;
  in >> n >> m;
  int per_thread = m / num_threads;
  Graph g(n, config, num_threads);
  std::vector<std::vector<GraphUpdate>> updates(num_threads,
          std::vector<GraphUpdate>(per_thread));

  int type;
  node_id_t a, b;
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < per_thread; ++j) {
      in >> type >> a >> b;
      updates[i][j] = {{a,b}, (UpdateType)type};
    }
  }
  for (edge_id_t i = per_thread * num_threads; i < m; ++i) {
    in >> type >> a >> b;
    g.update({{a,b}, (UpdateType)type});
  }

  auto task = [&updates, &g](int id) {
    for (auto upd : updates[id]) {
      g.update(upd, id);
    }
    return;
  };

  std::thread threads[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(task, i);
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

  g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
  g.connected_components();
}

TEST(GraphTest, MTStreamWithMultipleQueries) {
  for(int i = 1; i <= 10; i++) {
    auto config = GraphConfiguration().gutter_sys(STANDALONE);

    const std::string fname = __FILE__;
    size_t pos = fname.find_last_of("\\/");
    const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
    BinaryGraphStream_MT stream(curr_dir + "/res/multiples_graph_1024_stream.data", 256);
    BinaryGraphStream verify_stream(curr_dir + "/res/multiples_graph_1024_stream.data", 256);
    node_id_t num_nodes = verify_stream.nodes();
    edge_id_t num_edges = verify_stream.edges();
    MatGraphVerifier verify(num_nodes);

    int inserter_threads = i;
    std::vector<std::thread> threads;
    Graph g(num_nodes, config, inserter_threads);

    // variables for coordination between inserter_threads
    bool query_done = false;
    int num_query_ready = 0;
    std::condition_variable q_ready_cond;
    std::condition_variable q_done_cond;
    std::mutex q_lock;

    // prepare evenly spaced queries
    int num_queries = 10;
    int upd_per_query = num_edges / num_queries;
    int query_idx = upd_per_query;
    ASSERT_TRUE(stream.register_query(query_idx)); // register first query

    // task for threads that insert to the graph and perform queries
    auto task = [&](const int thr_id) {
      MT_StreamReader reader(stream);
      GraphUpdate upd;
      while(true) {
        upd = reader.get_edge();
        if (upd.second == END_OF_FILE) return;
        else if (upd.second == NXT_QUERY) {
          query_done = false;
          if (thr_id > 0) {
            // pause this thread and wait for query to be done
            std::unique_lock<std::mutex> lk(q_lock);
            num_query_ready++;
            lk.unlock();
            q_ready_cond.notify_one();

            // wait for query to finish
            lk.lock();
            q_done_cond.wait(lk, [&](){return query_done;});
            num_query_ready--;
            lk.unlock();
          } else {
            // this thread will actually perform the query
            // wait for other threads to be done applying updates
            std::unique_lock<std::mutex> lk(q_lock);
            num_query_ready++;
            q_ready_cond.wait(lk, [&](){
              return num_query_ready >= inserter_threads;
            });

            // add updates to verifier and perform query
            for (int j = 0; j < upd_per_query; j++) {
              GraphUpdate upd = verify_stream.get_edge();
              verify.edge_update(upd.first.first, upd.first.second);
            }
            verify.reset_cc_state();
            g.set_verifier(std::make_unique<MatGraphVerifier>(verify));
            g.connected_components(true);

            // inform other threads that we're ready to continue processing queries
            stream.post_query_resume();
            if(num_queries > 1) {
              // prepare next query
              query_idx += upd_per_query;
              ASSERT_TRUE(stream.register_query(query_idx));
              num_queries--;
            }
            num_query_ready--;
            query_done = true;
            lk.unlock();
            q_done_cond.notify_all();
          }
        }
        else if (upd.second == INSERT || upd.second == DELETE)
          g.update(upd, thr_id);
        else
          throw std::invalid_argument("Did not recognize edge code!");
      }
    };

    // start inserters
    for (int t = 0; t < inserter_threads; t++) {
      threads.emplace_back(task, t);
    }
    // wait for inserters to be done
    for (int t = 0; t < inserter_threads; t++) {
      threads[t].join();
    }

    // process the rest of the stream into the MatGraphVerifier
    for(size_t i = query_idx; i < num_edges; i++) {
      GraphUpdate upd = verify_stream.get_edge();
      verify.edge_update(upd.first.first, upd.first.second);
    }

    // perform final query
    std::cout << "Starting CC" << std::endl;
    verify.reset_cc_state();
    g.set_verifier(std::make_unique<MatGraphVerifier>(verify));
    ASSERT_EQ(g.connected_components().size(), 78);
  }
}
