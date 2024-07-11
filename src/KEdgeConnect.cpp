//
// Created by chen on 1/11/24.
//

#include "KEdgeConnect.h"

#include <algorithm>
#include <iostream>
#include <random>


KEdgeConnect::KEdgeConnect(node_id_t num_nodes, unsigned int num_forest, const std::vector<CCAlgConfiguration> &config_vec)
: num_nodes(num_nodes), num_forest(num_forest) {
    for(unsigned int i=0;i<num_forest;i++)
    {
        cc_alg.push_back(std::make_unique<CCSketchAlg>(num_nodes, config_vec[i]));
    }
}

KEdgeConnect::~KEdgeConnect() {
   //implement later
}

void KEdgeConnect::allocate_worker_memory(size_t num_workers) {
    for(unsigned int i=0;i<num_forest;i++){
        cc_alg[i]->allocate_worker_memory(num_workers);
    }
}

size_t KEdgeConnect::get_desired_updates_per_batch() {
    // I don't want to return double because the updates are sent to both
    // I copied from the two-edge-connect-class and did not understand 'updates are sent to both' -- Chen
    return cc_alg[0]->get_desired_updates_per_batch();
}

node_id_t KEdgeConnect::get_num_vertices() { return num_nodes; }

void KEdgeConnect::pre_insert(GraphUpdate upd, node_id_t thr_id) {
    for(unsigned int i=0;i<num_forest;i++) {
        cc_alg[i]->pre_insert(upd, thr_id);
    }
}

void KEdgeConnect::apply_update_batch(size_t thr_id, node_id_t src_vertex,
                        const std::vector<node_id_t> &dst_vertices) {
    for(unsigned int i=0;i<num_forest;i++) {
        cc_alg[i]->apply_update_batch(thr_id, src_vertex, dst_vertices);
    }
}

bool KEdgeConnect::has_cached_query() {
    bool cached_query_flag = true;
    for (unsigned int i=0;i<num_forest;i++) {
        cached_query_flag = cached_query_flag && cc_alg[i]->has_cached_query();
    }

    return cached_query_flag;
}

void KEdgeConnect::print_configuration() { cc_alg[0]->print_configuration(); }

void KEdgeConnect::query() {
    GraphUpdate temp_edge;
    temp_edge.type = DELETE;

    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> temp_forest;
    for(unsigned int i=0;i<num_forest-1;i++) {
        //std::cout << "SPANNING FOREST " << (i+1) << std::endl;
        // getting the spanning forest from the i-th cc-alg
        temp_forest = cc_alg[i]->calc_spanning_forest();
        forests_collection.push_back(temp_forest);

        for (unsigned int j = 0; j < temp_forest.size(); j++) {
            //std::cout << temp_forest[j].first << ":";
            for (auto dst: temp_forest[j].second) {
                //std::cout << " " << dst;
                temp_edge.edge.src = temp_forest[j].first;
                temp_edge.edge.dst = dst;
                for (int l=i+1;l<num_forest;l++){
                    cc_alg[l]->update(temp_edge);
                }
            }
            //std::cout << std::endl;
        }
    }

    //std::cout << "THE LAST SPANNING FOREST" << std::endl;
    temp_forest = cc_alg[num_forest-1]->calc_spanning_forest();
    forests_collection.push_back(temp_forest);
    for (unsigned int j = 0; j < temp_forest.size(); j++) {
        //std::cout << temp_forest[j].first << ":";
        for (auto dst: temp_forest[j].second) {
            //std::cout << " " << dst;
        }
        //std::cout << std::endl;
    }
}
