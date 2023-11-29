#include "binary_file_stream.h"


void generate_binary_stream(std::string filename, node_id_t num_nodes) {
    if (num_nodes % 2 == 1) {
        std::cerr << "Error, stream generator must have even number of nodes." << std::endl;
        return;
    }
    BinaryFileStream fout(filename, false);
    edge_id_t num_rounds = num_nodes/8;
    edge_id_t num_edges = num_nodes-2 + 2*num_rounds*num_nodes;
    fout.write_header(num_nodes, num_edges);
    // Build two large components
    GraphStreamUpdate update;
    update.type = INSERT;
    for (node_id_t u=0; u<num_nodes/2-1; u++) {
        update.edge = {u,u+1};
        fout.write_updates(&update, 1);
        update.edge = {u+num_nodes/2,u+1+num_nodes/2};
        fout.write_updates(&update, 1);
    }
    // Repeatedly add and remove edges between the two components num_rounds times
    for (int i = 0; i < num_rounds; i++) {
        std::cout << "GENERATING ROUND " << i << " OF " << num_rounds << " IN " << filename << std::endl;
        // First insert a bunch of edges across the cut and then delete them
        update.type = INSERT;
        for (node_id_t u=0; u<num_nodes/2; u++) {
            update.edge = {u,u+num_nodes/2};
            fout.write_updates(&update, 1);
        }
        update.type = DELETE;
        for (node_id_t u=0; u<num_nodes/2; u++) {
            update.edge = {u,u+num_nodes/2};
            fout.write_updates(&update, 1);
        }
        // Next repeatedly insert and delete one edge across the cut
        for (node_id_t u=0; u<num_nodes/2; u++) {
            update.type = INSERT;
            update.edge = {u,u+num_nodes/2};
            fout.write_updates(&update, 1);
            update.type = DELETE;
            update.edge = {u,u+num_nodes/2};
            fout.write_updates(&update, 1);
        }
    }
}

int main(int argc, char** argv) {
    generate_binary_stream("scut_13_stream_binary", 8192);
    return 0;
}
