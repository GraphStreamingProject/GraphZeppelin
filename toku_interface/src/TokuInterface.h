#ifndef TOKU_INTER_GUARD
#define TOKU_INTER_GUARD

#include <tokudb.h>
#include <vector>
#include <utility>
#include <unordered_map>

class TokuInterface {
public:
    TokuInterface();
    ~TokuInterface();

    bool putEdge(std::pair<uint64_t, uint64_t> edge, int8_t value);
    void flush();

    std::vector<std::pair<uint64_t, int8_t>>* getEdges(uint64_t node);
private:

    DB_ENV *env;
    DB *db;

    const int MAX_DB_FILENAME_LENGTH = 100;
	const int MAX_ENV_DIRNAME_LENGTH = 300;

    const char* DB_FILE = "graph-stream-file_v0.1";
    const char* DB_DIR = "graphDir";

    std::unordered_map<uint64_t, uint64_t> update_counts;
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
#endif