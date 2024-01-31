#include <fstream>
#include <map>
#include <vector>
#include <random>

#include "bucket.h"
#include "binary_file_stream.h"
#include "util.h"

void old_generate_binary_stream(std::string filename, node_id_t num_nodes, bool metis) {
    if (num_nodes % 2 == 1) {
        std::cerr << "Error, stream generator must have even number of nodes." << std::endl;
        return;
    }
    BinaryFileStream fout(filename, false);

    edge_id_t num_edges = num_nodes-2 + ((num_nodes/2) * (num_nodes/2));
    fout.write_header(num_nodes, num_edges);

    // Warning: Current design of writing to METIS stores all the edges, so this won't work with large number of edges          
    std::map<node_id_t, std::vector<node_id_t>> nodes_list;
    std::ofstream metis_file;
    if(metis) {
        std::cout << "Writing to METIS file has been enabled\n";
        metis_file.open(filename + ".metis");
        metis_file << num_nodes << " " << num_edges << " 0" << "\n";
    }

    // Build two large components
    std::cout << "Building two large components...\n";
    GraphStreamUpdate update;
    update.type = INSERT;
    for (node_id_t u=0; u<num_nodes/2-1; u++) {
        update.edge = {u,u+1};
        fout.write_updates(&update, 1);
        update.edge = {u+num_nodes/2,u+1+num_nodes/2};
        fout.write_updates(&update, 1);

        if(metis) {
            if (nodes_list.find(u) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u] = std::vector<node_id_t>();
            }
            if (nodes_list.find(u+1) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u+1] = std::vector<node_id_t>();
            }
            if (nodes_list.find(u+num_nodes/2) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u+num_nodes/2] = std::vector<node_id_t>();
            }
            if (nodes_list.find(u+1+num_nodes/2) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u+1+num_nodes/2] = std::vector<node_id_t>();
            }

            nodes_list[u].push_back(u+1);
            nodes_list[u+1].push_back(u);
            nodes_list[u+num_nodes/2].push_back(u+1+num_nodes/2);
            nodes_list[u+1+num_nodes/2].push_back(u+num_nodes/2);
        }
    }

    // Fully connect two large components
    std::cout << "Fully connecting two large components...\n";
    for (node_id_t src=0; src<num_nodes/2; src++) {
        if (src % 100 == 0) { // Print for every 100 nodes
            std::cout << "  Current node = " << src << "\n";
        }
        for (node_id_t dst=num_nodes/2; dst<num_nodes; dst++) {
            update.edge = {src,dst};
            fout.write_updates(&update, 1);

            if(metis) {
                if (nodes_list.find(src) == nodes_list.end()) { // Has not been inserted yet
                    nodes_list[src] = std::vector<node_id_t>();
                }
                if (nodes_list.find(dst) == nodes_list.end()) { // Has not been inserted yet
                    nodes_list[dst] = std::vector<node_id_t>();
                }

                nodes_list[src].push_back(dst);
                nodes_list[dst].push_back(src);
            }
        }
    }

    if(metis) { // Write to metis file
        std::cout << "Writing to METIS file...\n";
        for (auto it = nodes_list.begin(); it != nodes_list.end(); it++) {
            for (size_t neighbor = 0; neighbor < it->second.size() - 1; neighbor++) {
                metis_file << it->second[neighbor] + 1 << " ";
            }
            metis_file << it->second[it->second.size() - 1] + 1 << "\n";
        }
        metis_file.close();
    }
}

void generate_binary_stream(std::string filename, node_id_t num_nodes, bool metis) {
    if (num_nodes % 2 == 1) {
        std::cerr << "Error, stream generator must have even number of nodes." << std::endl;
        return;
    }
    BinaryFileStream fout(filename, false);

    uint64_t seed = 0;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<> left_dist(0, num_nodes / 2);
    std::uniform_int_distribution<> right_dist(num_nodes / 2, num_nodes);

    edge_id_t min_cut = 10;
    edge_id_t num_edges = num_nodes-2 + ((num_nodes/2) * (num_nodes/4) * 2) + min_cut;
    fout.write_header(num_nodes, num_edges);

    // Warning: Current design of writing to METIS stores all the edges, so this won't work with large number of edges          
    std::map<node_id_t, std::vector<node_id_t>> nodes_list;
    std::ofstream metis_file;
    if(metis) {
        std::cout << "Writing to METIS file has been enabled\n";
        metis_file.open(filename + ".metis");
        metis_file << num_nodes << " " << num_edges << " 0" << "\n";
    }

    // Build two connected components
    std::cout << "Building two connected components...\n";
    GraphStreamUpdate update;
    update.type = INSERT;
    for (node_id_t u=0; u<num_nodes/2-1; u++) {
        update.edge = {u,u+1};
        fout.write_updates(&update, 1);
        update.edge = {u+num_nodes/2,u+1+num_nodes/2};
        fout.write_updates(&update, 1);

        if(metis) {
            if (nodes_list.find(u) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u] = std::vector<node_id_t>();
            }
            if (nodes_list.find(u+1) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u+1] = std::vector<node_id_t>();
            }
            if (nodes_list.find(u+num_nodes/2) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u+num_nodes/2] = std::vector<node_id_t>();
            }
            if (nodes_list.find(u+1+num_nodes/2) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[u+1+num_nodes/2] = std::vector<node_id_t>();
            }

            nodes_list[u].push_back(u+1);
            nodes_list[u+1].push_back(u);
            nodes_list[u+num_nodes/2].push_back(u+1+num_nodes/2);
            nodes_list[u+1+num_nodes/2].push_back(u+num_nodes/2);
        }
    }

    // Add edges within each components
    std::cout << "Adding edges within two components...\n";
    for (node_id_t i = 0; i < (num_nodes/2); i++) {
        if (i % 100 == 0) { // Print for every 100 iterations
            std::cout << "  Current iteration = " << i << "\n";
        }
        for (node_id_t j = 0; j < (num_nodes/4); j++) {
            // Add edge to left component
            node_id_t left_node1 = left_dist(mt);
            node_id_t left_node2 = left_dist(mt);

            while (left_node1 == left_node2) { // If equal, keep rolling
                left_node2 = left_dist(mt);
            }
            update.edge = {left_node1, left_node2};
            fout.write_updates(&update, 1);

            // Add edge to right component
            node_id_t right_node1 = right_dist(mt);
            node_id_t right_node2 = right_dist(mt);

            while (right_node1 == right_node2) { // If equal, keep rolling
                right_node2 = right_dist(mt);
            }
            update.edge = {right_node1, right_node2};
            fout.write_updates(&update, 1);

            if(metis) {
                nodes_list[left_node1].push_back(left_node2);
                nodes_list[left_node2].push_back(left_node1);
                nodes_list[right_node1].push_back(right_node2);
                nodes_list[right_node2].push_back(right_node1);
            }
        }
    }


    // Add edges between two components, this will be the min cut value
    std::cout << "Adding edges that represent min-cut...\n";
    for (node_id_t i = 0; i < min_cut; i++) {
        node_id_t left_node = left_dist(mt);
        node_id_t right_node = right_dist(mt);

        update.edge = {left_node, right_node};
        fout.write_updates(&update, 1);

        std::cout << "  Min cut edge " << i << ": {" << left_node << ", " << right_node << "}\n";  

        if(metis) {
            /*if (nodes_list.find(left_node1) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[left_node1] = std::vector<node_id_t>();
            }
            if (nodes_list.find(left_node2) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[left_node1] = std::vector<node_id_t>();
            }
            if (nodes_list.find(right_node1) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[right_node1] = std::vector<node_id_t>();
            }
            if (nodes_list.find(right_node2) == nodes_list.end()) { // Has not been inserted yet
                nodes_list[right_node1] = std::vector<node_id_t>();
            }*/

            nodes_list[left_node].push_back(right_node);
            nodes_list[right_node].push_back(left_node);
        }
    }


    if(metis) { // Write to metis file
        std::cout << "Writing to METIS file...\n";
        for (auto it = nodes_list.begin(); it != nodes_list.end(); it++) {
            for (size_t neighbor = 0; neighbor < it->second.size() - 1; neighbor++) {
                metis_file << it->second[neighbor] + 1 << " ";
            }
            metis_file << it->second[it->second.size() - 1] + 1 << "\n";
        }
        metis_file.close();
    }
}

int main(int argc, char** argv) {
    size_t num_nodes = 1024;
    size_t num_max_edges = (num_nodes * (num_nodes - 1)) / 2;
    for (int i = 0; i < 10; i++) {
        size_t hash_value = Bucket_Boruvka::get_index_hash(i, 1);
        Edge edge = inv_concat_pairing_fn(hash_value);

        size_t edge_id = hash_value % num_max_edges;

        //std::cout << "Hash value: " << hash_value + 1000000000 << " Edge: {" << edge.src << ", " << edge.dst << "}\n";
        std::cout << "Edge ID: " << edge_id << " Edge: {" << "\n";
    }
    //generate_binary_stream("nt_10_stream_binary", 1024, true);
    return 0;
}