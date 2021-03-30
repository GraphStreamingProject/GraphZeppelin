#ifndef TOKU_INTER_GUARD
#define TOKU_INTER_GUARD

#include <tokudb.h>
#include <vector>
#include <utility>
#include <unordered_map>
#include <atomic>

// forward declaration for co-dependency
class Graph;

class TokuInterface {
public:
    TokuInterface();
    ~TokuInterface();

    bool putEdge(std::pair<uint64_t, uint64_t> edge);
    std::vector<uint64_t> getEdges(uint64_t node);
    void flush();

    // pointer to the array of supernodes modified by putEdge() and flush()
    Graph* graph;
    
private:
    DB_ENV *env;
    DB *db;
    std::unordered_map<uint64_t, std::atomic<uint64_t>> update_counts;

    static const int MAX_DB_FILENAME_LENGTH = 100;
	static const int MAX_ENV_DIRNAME_LENGTH = 300;
    const char* DB_FILE = "graph-stream-file_v0.1";
    const char* DB_DIR = "graphDir";

    bool putSingleEdge(uint64_t src, uint64_t dst, int8_t val);

    // functions to extract relevant info from the dbt
    static inline uint64_t getNode(DBT *dbt) {
        return *(uint64_t *) dbt->data;
    }
    static inline uint64_t getEdgeTo(DBT *dbt) {
        return *(uint64_t *) ((uint8_t *)dbt->data + sizeof(uint64_t));
    }
    static inline int8_t getValue(DBT *dbt) {
        if (dbt->data == nullptr) return 0;
        return * (int8_t *) dbt->data;
    }
};

// int keyCompare(DB* db __attribute__((__unused__)), const DBT *a, const DBT *b);
// DBT *toDBT(uint64_t src, uint64_t dst, uint64_t rand=1);
#endif
