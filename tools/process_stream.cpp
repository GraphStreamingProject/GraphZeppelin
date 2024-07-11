#include <binary_file_stream.h>
#include <cc_sketch_alg.h>
#include <KEdgeConnect.h>
#include <graph_sketch_driver.h>
#include <sys/resource.h>  // for rusage
#include <test/mat_graph_verifier.h>
#include <map>
#include <xxhash.h>
#include <vector>
#include <algorithm>

#include <thread>

// TODO: make num_edge_connect an input argument;
// TODO: Daniel's concern: right now, the deletion of the edge to the stream makes the query only possible to be supported once -- fix this later

int ctr_x=0;
int ctr_y=0;

unsigned long long ctr_z=0;
unsigned long long ctr_o=0;
unsigned long long ctr_t=0;


static bool shutdown = false;

class MinCutSimple {
public:
    const node_id_t num_nodes;
    unsigned int num_forest;  // this value is k in the k-edge connectivity
    unsigned int num_subgraphs;
    const int my_prime = 44497; // Prime number for polynomial hashing
    std::vector<std::vector<int>> hash_coefficients;
    std::vector<std::unique_ptr<KEdgeConnect>> k_edge_algs;
    std::vector<unsigned int> mincut_values;
    unsigned int return_min_cut;

    explicit MinCutSimple(node_id_t num_nodes, const std::vector<std::vector<CCAlgConfiguration>> &config_vec):
            num_nodes(num_nodes) {
        num_subgraphs = (unsigned int)(2*std::ceil(std::log2(num_nodes)));
  	std::cout<<"Number of subgraphs(constructor):"<<num_subgraphs<<std::endl;
        // TODO: make the approximation factor tunable later
        num_forest = 2*num_subgraphs;
  	std::cout<<"Number of forests(k)(constructor):"<<num_forest<<std::endl;
        for(unsigned int i=0;i<num_subgraphs;i++){
            k_edge_algs.push_back(std::make_unique<KEdgeConnect>(num_nodes, num_forest, config_vec[i]));
        }
        // Initialize coefficients randomly
  	//std::cout<<"hash coeffs (constructor):"<<std::endl;
        std::random_device rd_ind;
        std::mt19937 gen_ind(rd_ind());
        std::uniform_int_distribution<int> dist_coeff(1, my_prime - 1); // random numbers between 1 and p-1
        for (int i =0; i<num_subgraphs; i++) {
            std::vector<int> this_subgraph_coeff;
            for (int j = 0; j < num_subgraphs; j++) {
                this_subgraph_coeff.push_back(dist_coeff(gen_ind));
  		//std::cout<<this_subgraph_coeff[j]<<" ";
            }
            hash_coefficients.push_back(this_subgraph_coeff);
	    std::cout<<std::endl;
        }
  	std::cout<<"end of constructor"<<std::endl;
    }

    void allocate_worker_memory(size_t num_workers){
        for(unsigned int i=0;i<num_subgraphs;i++){
            k_edge_algs[i]->allocate_worker_memory(num_workers);
        }
    }

    size_t get_desired_updates_per_batch(){
        return k_edge_algs[0]->get_desired_updates_per_batch();
    }

    node_id_t get_num_vertices() { return num_nodes; }

    // Function to calculate power modulo prime
    static int power(int x, int y, int p) {
        int res = 1; // Initialize result
        x = x % p; // Update x if it is more than or equal to p
        while (y > 0) {
            // If y is odd, multiply x with result
            if (y & 1)
                res = (res * x) % p;
            // y must be even now
            y = y >> 1; // y = y/2
            x = (x * x) % p;
        }
        return res;
    }

    // Function to generate k-wise independent hash
    int k_wise_hash(const std::vector<int>& coefficients, unsigned int src_vertex, unsigned int dst_vertex) {
        unsigned int hash_val = 0;
        if (src_vertex>dst_vertex){
            std::swap(src_vertex, dst_vertex);
        }
        unsigned int edge_id = src_vertex*num_nodes + dst_vertex;
        for (int i = 0; i < coefficients.size(); ++i) {
            hash_val = (hash_val + coefficients[i] * power(edge_id, i, my_prime)) % my_prime;
        }
	//std::cout<<"Hash val inside hash func: "<<(hash_val % 2)<<std::endl;
	if(hash_val % 2==0)
	{
		ctr_x++;
	}
	else
	{
		ctr_y++;
	}

        return (hash_val % 2);
    }

    
    void pre_insert(GraphUpdate upd, node_id_t thr_id) {
        for(unsigned int i=0;i<num_subgraphs;i++){
            k_edge_algs[i]->pre_insert(upd, thr_id);
        }
    }

    // Custom comparator function to sort dst_vertices based on end_index in descending order
    static bool compareEndIndexDescending(const unsigned int &a, const unsigned int &b, const std::vector<unsigned int> &end_index) {
        return end_index[a] > end_index[b];
    }


    void apply_update_batch(size_t thr_id, node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices) {
        /************* tests
        for(unsigned int i=0;i<num_subgraphs;i++) {
                k_edge_algs[i]->apply_update_batch(thr_id, src_vertex, dst_vertices);
            }
        ***************/
        std::vector<std::pair<unsigned int, unsigned int>> dst_end_index;
        // Collect the end-index on which an edge is deleted, then sort the indices to achieve O(N) total update time
        // for the vector.
        std::pair<node_id_t, unsigned int> temp_pair;
        for (auto dst_vertex: dst_vertices){
            for (unsigned int i=1;i<num_subgraphs;i++) {
                if(k_wise_hash(hash_coefficients[i], src_vertex, dst_vertex)==0) {
                    temp_pair.first = dst_vertex;
                    temp_pair.second = i;
                    dst_end_index.push_back(temp_pair);
                    break;
                }
            }
        }
        // std::cout<<"We made it outside the for loop!"<<std::endl;
        // Sort end_index vector by the end_index of dst_end_index
        std::sort(dst_end_index.begin(), dst_end_index.end(), [](auto &left, auto &right) {
            return left.second > right.second;
        });

        std::vector<node_id_t> input_dst_vertices;
        for (auto & pair : dst_end_index){
            input_dst_vertices.push_back(pair.first);
        }
        int position;
        if (!input_dst_vertices.empty()) {
            position = input_dst_vertices.size() - 1;
        }
        else{
            position = -1;
        }
	
	//std::cout<<"Vertex: "<<src_vertex<<std::endl;
        for(unsigned int i=0;i<num_subgraphs;i++) {
            if (position>=0) {
                k_edge_algs[i]->apply_update_batch(thr_id, src_vertex, input_dst_vertices);
		//std::cout<<"Nbhd size in round "<<i<<" is: "<<input_dst_vertices.size()<<std::endl;
		if(i==0)
		{
			ctr_z = ctr_z + input_dst_vertices.size();
		}
		else if(i==1)
		{
			ctr_o = ctr_o + input_dst_vertices.size();
		}
		else if(i==2)
		{
			ctr_t = ctr_t + input_dst_vertices.size();
		}
                // The following while loop: keep the position variable to be always aligned with the last vertex
                // that has not been deleted yet
                // If the vertex-corresponding end_index is at most the iteration number, it means it has already been
                // accounted for in this iteration, and should not be accounted of at the next iteration; so we should
                // remove
                while (dst_end_index[position].second <= i+1 && position>=0) {
                    input_dst_vertices.pop_back();
                    position--;
                }
            }
        }
    }

    bool has_cached_query(){
	return false;
        // bool cached_query_flag = true;
        // for (unsigned int i=0;i<num_subgraphs;i++) {
        //     cached_query_flag = cached_query_flag && k_edge_algs[i]->has_cached_query();
        // }
    }

    void print_configuration(){k_edge_algs[0]->print_configuration(); }


    void write_k_edge_cert(std::map<size_t, std::vector<size_t>>& nodes_list, size_t num_edge){
        // below are the codes stolen from the converter file
        std::string file_name = "temp-graph-min-cut.metis";
        std::ofstream metis_file(file_name);

        std::cout << "Writing METIS file...\n";
	std::cout<<"no of vertices and edges "<< num_nodes << " " << num_edge << std::endl;

        // could be a hidden bug later -- at the moment, num_nodes is taken from the class
        metis_file << num_nodes << " " << num_edge << " 0" << "\n";

        for (unsigned int it=0; it<num_nodes;it++) {
            for (size_t neighbor = 0; neighbor < nodes_list[it].size(); neighbor++) {
                if (nodes_list[it][neighbor] == it) {
                    continue;
                }
                metis_file << (1+nodes_list[it][neighbor]) << " ";

            }
            metis_file << "\n";
        }

        metis_file.close();

        std::cout << "Finished Writing METIS file...\n";

	int new_count=0;
        for (unsigned int it=0; it<num_nodes;it++) {
            for (size_t neighbor = 0; neighbor < nodes_list[it].size(); neighbor++) {
                if (nodes_list[it][neighbor] == it) {
                    continue;
		}
		new_count++;
	    }
	}
	std::cout<<"no of edges (going through the nodelist): "<<new_count<<std::endl;
     }

    void query(){
        std::cout<<std::endl<<"We made it to the query function!"<<std::endl;
        for(unsigned int i=0;i<num_subgraphs;i++) {
            std::map<size_t, std::vector<size_t>> nodes_list;
            std::vector<std::pair<node_id_t, std::vector<node_id_t>>> current_forest;
            size_t subgraph_num_edge = 0;
            // easy version: check from the i=0 to i=log n without using an additional layer of binary search
            k_edge_algs[i]->query(); // This creates the k forests
            unsigned int k = k_edge_algs[i]->forests_collection.size();
	    std::cout<<"query func, k= "<<k<<std::endl;
            for (unsigned int j=0;j<k;j++) {
                current_forest = k_edge_algs[i]->forests_collection[j];
                for(auto v_neighbors_pair: current_forest){
                    for (auto neighbor_ver: v_neighbors_pair.second){
                        nodes_list[v_neighbors_pair.first].push_back(neighbor_ver);
                        nodes_list[neighbor_ver].push_back(v_neighbors_pair.first);
                        subgraph_num_edge++;
                    }
                }
            }
            write_k_edge_cert(nodes_list, subgraph_num_edge);
            // run the min-cut algorithm -- stolen from gpu min-cut
            std::string file_name = "temp-graph-min-cut.metis";
            std::string output_name = "mincut.txt";
            std::string command = "./mincut_parallel " + file_name + " exact >" + output_name; // Run VieCut and store the output
            std::system(command.data());

            std::string line;
            std::ifstream output_file(output_name);
            if (output_file.is_open()) {
                while (std::getline(output_file, line)) {
                    size_t cut_pos = line.find("cut=");
                    if (cut_pos != std::string::npos) {
                        unsigned int cut_value = std::stoul(line.substr(cut_pos + 4));
                        std::cout << "Cut value: " << cut_value << std::endl;
                        mincut_values.push_back(cut_value);
                        break; // Stop reading after finding the first "cut=" value
                    }
                }
                output_file.close();
            } else {
                std::cout << "Error: Couldn't find file name: " << output_name << "!\n";
            }
            if (mincut_values[i]<num_forest){
                return_min_cut = mincut_values[i] * std::pow(2, i);
                break;
            }
        }
    }

};
// class

class TwoEdgeConnect {
 public:
  const node_id_t num_nodes;
  CCSketchAlg cc_alg_1;
  CCSketchAlg cc_alg_2;

  explicit TwoEdgeConnect(node_id_t num_nodes, const CCAlgConfiguration &config_1,
                          const CCAlgConfiguration &config_2)
      : num_nodes(num_nodes), cc_alg_1(num_nodes, config_1), cc_alg_2(num_nodes, config_2) {}

  void allocate_worker_memory(size_t num_workers) {
    cc_alg_1.allocate_worker_memory(num_workers);
    cc_alg_2.allocate_worker_memory(num_workers);
  }

  size_t get_desired_updates_per_batch() {
    // I don't want to return double because the updates are sent to both
    return cc_alg_1.get_desired_updates_per_batch();
  }

  node_id_t get_num_vertices() { return num_nodes; }

  void pre_insert(GraphUpdate upd, node_id_t thr_id) {
    cc_alg_1.pre_insert(upd, thr_id);
    cc_alg_2.pre_insert(upd, thr_id);
  }

  void apply_update_batch(size_t thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices) {
    cc_alg_1.apply_update_batch(thr_id, src_vertex, dst_vertices);
    cc_alg_2.apply_update_batch(thr_id, src_vertex, dst_vertices);
  }

  bool has_cached_query() { return cc_alg_1.has_cached_query() && cc_alg_2.has_cached_query(); }

  void print_configuration() { cc_alg_1.print_configuration(); }

  void query() {
    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest =
        cc_alg_1.calc_spanning_forest();

    GraphUpdate temp_edge;

    temp_edge.type = DELETE;

    std::cout << "SPANNING FOREST 1" << std::endl;
    for (unsigned int j = 0; j < forest.size(); j++) {
      std::cout << forest[j].first << ":";
      for (auto dst : forest[j].second) {
        std::cout << " " << dst;
        temp_edge.edge.src = forest[j].first;
        temp_edge.edge.dst = dst;
        cc_alg_2.update(temp_edge);
      }
      std::cout << std::endl;
    }

    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest2 =
        cc_alg_2.calc_spanning_forest();

    std::cout << "SPANNING FOREST 2" << std::endl;
    for (unsigned int j = 0; j < forest.size(); j++) {
        std::cout << forest[j].first << ":";
        for (auto dst: forest[j].second) {
            std::cout << " " << dst;
        }
        std::cout << std::endl;
    }

    // TODO: reinsert into alg 2?
  }
};

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double)data.ru_maxrss / 1024.0;
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
template <typename DriverType>

void track_insertions(uint64_t total, DriverType *driver,
                      std::chrono::steady_clock::time_point start_time) {
  total = total * 2;  // we insert 2 edge updates per edge

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r");
  fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev = start_time;
  uint64_t prev_updates = 0;

  while (true) {
    sleep(1);
    uint64_t updates = driver->get_total_updates();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff = now - prev;

    // calculate the insertion rate
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    size_t ins_per_sec = (((double)(upd_delta)) / cur_diff.count()) / 2;

    if (updates >= total || shutdown) break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(),
           std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %lu per second\r", progress * 5, ins_per_sec);
    fflush(stdout);
  }

  printf("Progress:====================| Done                             \n");
  return;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  shutdown = false;
  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  size_t reader_threads = std::atol(argv[3]);

  unsigned int num_subgraphs;
  unsigned int num_edge_connect;

  BinaryFileStream stream(stream_file);
  BinaryFileStream stream_ref(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  num_subgraphs = (unsigned int)(2*std::ceil(std::log2(num_nodes)));
  num_edge_connect = 2*num_subgraphs;
  std::cout<<"Number of subgraphs:"<<num_subgraphs<<std::endl;
  std::cout<<"Edges Connectivity Param (k):"<<num_edge_connect<<std::endl;

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  std::vector<std::vector<CCAlgConfiguration>> config_vec;

  for (unsigned int i=0;i<num_subgraphs;i++){
      std::vector<CCAlgConfiguration> subgraph_config_vec;
      for (unsigned int j=0;j<num_edge_connect;j++){
          subgraph_config_vec.push_back(CCAlgConfiguration().batch_factor(1));
      }
      config_vec.push_back(subgraph_config_vec);
  }

  // KEdgeConnect k_edge_alg{num_nodes, num_edge_connect, config_vec};
  MinCutSimple min_cut_alg{num_nodes, config_vec};

  GraphSketchDriver<MinCutSimple> driver{&min_cut_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions<GraphSketchDriver<MinCutSimple>>, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);

  auto cc_start = std::chrono::steady_clock::now();
  driver.prep_query();

  min_cut_alg.query();


//  size_t m = stream_ref.edges();
//  // test the edges in the spanning forest are in the original graph
//  MatGraphVerifier kEdgeVerifier(num_nodes);
//
//  while (m--) {
//     GraphStreamUpdate upd;
//     stream_ref.get_update_buffer(&upd, 1);
//     kEdgeVerifier.edge_update(upd.edge.src, upd.edge.dst);
//   }
//
//  std::vector<std::vector<bool>> test_adj_mat(num_nodes);
//  test_adj_mat =  kEdgeVerifier.extract_adj_matrix();
//
//  Edge temp_edge;
//  std::vector<std::pair<node_id_t, std::vector<node_id_t>>> temp_forest;
//  for(unsigned int i=0;i<num_edge_connect;i++) {
//      temp_forest = k_edge_alg.forests_collection[i];
//      // Test the maximality of the connected components
//      DisjointSetUnion<node_id_t> kruskal_dsu(num_nodes);
//      std::vector<std::set<node_id_t>> temp_retval;
//      for (unsigned int l = 0; l < temp_forest.size(); l++) {
//          for (unsigned int j = 0; j < temp_forest[l].second.size(); j++) {
//              kruskal_dsu.merge(temp_forest[l].first, temp_forest[l].second[j]);
//          }
//      }
//      std::map<node_id_t, std::set<node_id_t>> temp_map;
//      for (unsigned l = 0; l < num_nodes; ++l) {
//          temp_map[kruskal_dsu.find_root(l)].insert(l);
//      }
//      temp_retval.reserve(temp_map.size());
//      for (const auto& entry : temp_map) {
//          temp_retval.push_back(entry.second);
//      }
//      std::cout<< std::endl;
//      kEdgeVerifier.reset_cc_state();
//      kEdgeVerifier.verify_soln(temp_retval);
//      // End of the test of CC maximality
//      // start of the test of CC edge existence
//      for (unsigned int j = 0; j < temp_forest.size(); j++) {
//            for (auto dst: temp_forest[j].second) {
//                temp_edge.src = temp_forest[j].first;
//                temp_edge.dst = dst;
//                kEdgeVerifier.verify_edge(temp_edge);
//                kEdgeVerifier.edge_update(temp_edge.src, temp_edge.dst);
//            }
//      }
//      test_adj_mat =  kEdgeVerifier.extract_adj_matrix();
//  }

//  unsigned long CC_nums[num_edge_connect];
//  for(unsigned int i=0;i<num_edge_connect;i++){
//      CC_nums[i]= k_edge_alg.cc_alg[i]->connected_components().size();
//  }

  std::chrono::duration<double> insert_time = driver.flush_end - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  std::chrono::duration<double> flush_time = driver.flush_end - driver.flush_start;
  std::chrono::duration<double> cc_alg_time =
          min_cut_alg.k_edge_algs[num_subgraphs-1]->cc_alg[num_edge_connect-1]->cc_alg_end - min_cut_alg.k_edge_algs[0]->cc_alg[0]->cc_alg_start;

  shutdown = true;
  querier.join();

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec):    " << num_seconds << std::endl;
  std::cout << "Updates per second:           " << stream.edges() / num_seconds << std::endl;
  std::cout << "Total CC query latency:       " << cc_time.count() << std::endl;
  std::cout << "  Flush Gutters(sec):         " << flush_time.count() << std::endl;
//  std::cout << "  Boruvka's Algorithm(sec):     " << cc_alg_time.count() << std::endl;
  std::cout << "Min-cut size:                 " << min_cut_alg.return_min_cut << std::endl;
//  for(unsigned int i=0;i<num_edge_connect;i++){
//      std::cout << "Number of connected Component in :         " << i+1 << " is " << CC_nums[i] << std::endl;
//  }
  std::cout << "Maximum Memory Usage(MiB):    " << get_max_mem_used() << std::endl;


  std::cout<<"No of times hash func returns 0 (x)="<<ctr_x<<std::endl;
  std::cout<<"No of times hash func returns 1 (y)="<<ctr_y<<std::endl;
  std::cout<<"Counter to check numbers in sampling using concentration"<<std::endl;
  std::cout<<"count of 0: "<<ctr_z<<std::endl;
  std::cout<<"count of 1: "<<ctr_o<<std::endl;
  std::cout<<"count of 2: "<<ctr_t<<std::endl;
}
