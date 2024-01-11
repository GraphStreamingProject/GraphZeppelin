//
// Created by chen on 1/11/24.
//

#pragma once
#include <atomic>  // REMOVE LATER
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <unordered_set>
#include <vector>
#include <memory>
#include <cassert>
#include <cc_sketch_alg.h>

#include "cc_alg_configuration.h"
#include "sketch.h"
#include "dsu.h"


class KEdgeConnect {
public:
    const node_id_t num_nodes;
    const unsigned int num_forest;
    std::vector<std::unique_ptr<CCSketchAlg>> cc_alg;

    explicit KEdgeConnect(node_id_t num_nodes, unsigned int num_forest, const std::vector<CCAlgConfiguration> &config_vec);
    ~KEdgeConnect();

    void allocate_worker_memory(size_t num_workers);

    size_t get_desired_updates_per_batch();

    node_id_t get_num_vertices();

    void pre_insert(GraphUpdate upd, node_id_t thr_id);

    void apply_update_batch(size_t thr_id, node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices);

    bool has_cached_query();

    void print_configuration();

    void query();
};

