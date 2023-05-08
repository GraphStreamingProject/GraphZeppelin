#pragma once
#include <vector>
#include <map>
#include <util.h>

class CudaGraph {
    public: 
        // List of edge ids that thread will be responsble for updating
        vec_t *edgeUpdates;

        // List of num updates for each node
        int *nodeNumUpdates;

        // List of starting index for each node's update
        vec_t *nodeStartIndex;

        // List of storing updates from graph_worker
        std::map<int, std::vector<vec_t>> graphUpdates;

        std::vector<std::mutex> mutexes;

        int num_nodes;

        bool isInit = false;

        // Default constructor
        CudaGraph() {}

        void configure(vec_t* _edgeUpdates, int* _nodeNumUpdates, vec_t* _nodeStartIndex, int _num_nodes) {
            edgeUpdates = _edgeUpdates;
            nodeNumUpdates = _nodeNumUpdates;
            nodeStartIndex = _nodeStartIndex;  
            num_nodes = _num_nodes;

            for(int i = 0; i < num_nodes; i++) {
                graphUpdates[i] = std::vector<vec_t>{};            
            }
            mutexes = std::vector<std::mutex>(num_nodes);


            isInit = true;
        };

        void batch_update(node_id_t src, const std::vector<node_id_t> &edges) {
            if (!isInit) {
                std::cout << "CudaGraph has not been initialized!\n";
            }
            std::unique_lock<std::mutex> lk(mutexes[src]);
            for (const auto& edge : edges) {
                if (src < edge) {
                    graphUpdates[src].push_back(static_cast<vec_t>(concat_pairing_fn(src, edge)));
                } 
                else {
                    graphUpdates[src].push_back(static_cast<vec_t>(concat_pairing_fn(edge, src)));
                }
            }
            lk.unlock();
        };

        void fillParam() {
            vec_t nodeIt = 0;
            vec_t startIndex = 0;
            for (auto it = graphUpdates.begin(); it != graphUpdates.end(); it++) {
                nodeStartIndex[it->first] = startIndex;
                nodeNumUpdates[it->first] = it->second.size();

                for (int i = 0; i < it->second.size(); i++) {
                    edgeUpdates[nodeIt] = it->second.at(i);
                    nodeIt++;
                }
                startIndex += it->second.size();
            }
        }
};