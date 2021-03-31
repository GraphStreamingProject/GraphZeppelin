#include "../../include/TokuInterface.h"
#include "../../include/graph.h"
#include "../../include/GraphWorker.h"

#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <exception>
#include <stdexcept>
#include <chrono>

#define MB (uint64_t) 1 << 20

// Defines which allow different db directories to be chosen
#define USE_DEFAULT false // use default dbdir
#define NEW_DB_DIR "../graph-db-data" // rel path to alternate dbdir

// Define the threshold at which we do a query
#define TAU (uint32_t) 10000

// Define toku params (0 indicates default)
// TODO: move these to a config file
#define CACHESIZE  8                  // The RAM cache which toku mintains should be about half of available RAM (provided to toku in GBs)
#define BUFFERSIZE (uint64_t) MB << 3 // The size of each buffer each node maintains. Larger = faster insertion but slower query (4~64MB scale)
#define FANOUT     0                  // Number of children of each node (don't know if we want to touch this)
#define REDZONE    1                  // Percent of disk available when toku will cease to insert (ie 95% full if redzone is 5%)

// function to compare the data stored in the tree for sorting and querying
inline int keyCompare(DB* db __attribute__((__unused__)), const DBT *a, const DBT *b) {
    return memcmp(a->data, b->data, a->size);
}

// function to create a DBT from key info.
// if rand is passed in then we set the last 8 bytes to it's value
// if not then we set it to the current clock this serves as 'random' noise
// to make the keys unique
inline DBT *toDBT(uint64_t src, uint64_t dst, uint32_t rand=1) {
    DBT *key_dbt = new DBT();
    key_dbt->flags=0;
    key_dbt->size = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t);
    key_dbt->ulen = key_dbt->size;
    key_dbt->data = calloc(key_dbt->size, sizeof(uint8_t));

    // data must be in big-endian for this to work
    src = htobe64(src);
    dst = htobe64(dst);

    memcpy(key_dbt->data, &src, sizeof(uint64_t));
    memcpy((uint8_t *)key_dbt->data + sizeof(uint64_t), &dst, sizeof(uint64_t));
    if (rand == 1) {
        uint32_t uid = htobe32(std::chrono::system_clock::now().time_since_epoch().count());
        memcpy((uint8_t *)key_dbt->data + 2*sizeof(uint64_t), &uid, sizeof(uint32_t));
    } else {
        rand = htobe32(rand);
        memcpy((uint8_t *)key_dbt->data + 2*sizeof(uint64_t), &rand, sizeof(uint32_t));
    }
    
    return key_dbt;
}

void freeDBT(DBT *dbt) {
    free(dbt->data);
    delete dbt;
}

// this function combines the values of two identical keys by adding them
// some params are commented because they are unused
// TODO: maybe figure out a way to use this at some point
// int edge_update(DB */*db*/, const DBT */*key*/, const DBT *old_val, const DBT *extra,
//   void (*set_val)(const DBT *new_val, void *set_extra),
//   void *set_extra) {
//     int8_t new_val = TokuInterface::getValue((DBT *)old_val) + TokuInterface::getValue((DBT *)extra);
    
//     DBT *value_dbt = new DBT();
//     value_dbt->flags=0;
//     value_dbt->size = sizeof(int8_t);
//     value_dbt->ulen = sizeof(int8_t);
//     value_dbt->data = malloc(sizeof(int8_t));
//     memset(value_dbt->data, new_val, sizeof(int8_t));

//     set_val(value_dbt, set_extra);
//     return 0;
// }

TokuInterface::TokuInterface() {
    int err;
    char* dbfile = (char*) calloc(MAX_DB_FILENAME_LENGTH, sizeof(char));
    char* dbdir = (char*) calloc(MAX_ENV_DIRNAME_LENGTH, sizeof(char));// = {'\0'};

    strncpy(dbfile, DB_FILE, MAX_DB_FILENAME_LENGTH);
    dbfile[MAX_DB_FILENAME_LENGTH - 1] = '\0';

    strncpy(dbdir, DB_DIR, MAX_ENV_DIRNAME_LENGTH);
    dbdir[MAX_ENV_DIRNAME_LENGTH - 1] = '\0';


    int dbFlags = DB_CREATE;
    int envFlags = DB_PRIVATE|DB_INIT_MPOOL|DB_INIT_LOCK|DB_THREAD|DB_CREATE;

    db_env_set_direct_io(false); // I wonder if we should try using direct IO... might be better with big cache

    if (!USE_DEFAULT) {
        printf("setting to new db dir\n");
        strncpy(dbdir, NEW_DB_DIR, MAX_ENV_DIRNAME_LENGTH);
        dbdir[MAX_ENV_DIRNAME_LENGTH - 1] = '\0';
    }

    if(mkdir(dbdir, S_IRWXU|S_IRWXG) != 0 && errno != EEXIST) { // open dbdir
        printf("failed to create db dir %s\n", dbdir);
        exit(EXIT_FAILURE);
    }
    if (db_env_create(&env, 0) != 0) { // create db env
        printf("Failed to create db env!\n");
        exit(EXIT_FAILURE);
    }

    printf("Successfully created env\n");

    env->set_errfile(env, stderr);

    // set the function toku will use to compare data
    if (env->set_default_bt_compare(env, keyCompare) != 0) {
        printf("Failed to set compare func!\n");
        exit(EXIT_FAILURE);
    }

    printf("Successfully set errfile and compare func\n");

    // set cachesize
    if (CACHESIZE != 0) {
        printf("Setting toku cache size to %iGBs\n", CACHESIZE);
        if (env->set_cachesize(env, CACHESIZE, 0, 1) != 0) {
            printf("ERROR: failed to set cache size\n");
            exit(EXIT_FAILURE);
        }
    }
    // set redzone
    if (REDZONE != 0) {
        printf("Setting toku redzone to %i\n", REDZONE);
        if ((err = env->set_redzone(env, REDZONE)) != 0) {
            printf("ERROR: failed to set red zone: %d {%s}\n", err, db_strerror(err));
            exit(EXIT_FAILURE);
        }
    }
    printf("Opening toku environment\n");
    if (env->open(env, dbdir, envFlags, S_IRWXU|S_IRWXG) != 0) {
        printf("Failed to open env!\n");
        exit(EXIT_FAILURE);
    }

    printf("Successfully opened env\n");

    if (db_create(&db, env, 0) != 0) {
        printf("error creating db.");
        exit(EXIT_FAILURE);
    }

    // set buffersize
    if (BUFFERSIZE != 0) {
        printf("Setting toku buffer size to %lu\n", BUFFERSIZE);
        if ((err = db->set_pagesize(db, BUFFERSIZE)) != 0) {
            printf("ERROR: failed to set buffer size\n");
            exit(EXIT_FAILURE);
        }
    }
    // set fanout
    if (FANOUT != 0) {
        printf("Setting toku fanout to %i\n", FANOUT);
        if((err = db->set_fanout(db, FANOUT)) != 0) {
            printf("ERROR: failed to set fanout\n");
            exit(EXIT_FAILURE);
        }
    }

    if (db->open(db, NULL, dbfile, NULL, DB_BTREE, dbFlags, S_IRUSR|S_IWUSR|S_IRGRP) != 0) {
        printf("Error opening db!\n");
        exit(EXIT_FAILURE);
    }
    free(dbfile);
    free(dbdir);

    // create map which will store the number of unqueries updates
    // to the datastructure
    update_counts = std::unordered_map<uint64_t, std::atomic<uint64_t>>();
    printf("Finished creating TokuInterface\n");
}

TokuInterface::~TokuInterface() {
    if (db->close(db, 0) != 0) {
        printf("ERROR: failed to close db\n");
        exit(EXIT_FAILURE);
    }

    if (env->close(env, 0) != 0) {
        printf("ERROR: failed to close env\n");
        exit(EXIT_FAILURE);
    }
}

// part of the API
// this function inserts 2 edges into the db (one revereses the edge)
bool TokuInterface::putEdge(std::pair<uint64_t, uint64_t> edge) {
    if (!putSingleEdge(edge.first, edge.second, 1)) 
        return false;
    return putSingleEdge(edge.second, edge.first, 1);
}

bool TokuInterface::putSingleEdge(uint64_t src, uint64_t dst, int8_t val) {
    DBT value_dbt;
    value_dbt.flags=0;
    value_dbt.size = sizeof(int8_t);
    value_dbt.ulen = sizeof(int8_t);
    value_dbt.data = malloc(sizeof(int8_t));
    memset(value_dbt.data, val, sizeof(int8_t));
    DBT *toInsert = toDBT(src, dst);
    // printf("inserting %lu %lu %i\n", src, dst, val);
    if (db->put(db, NULL, toInsert, &value_dbt, 0) != 0) {
        printf("ERROR: failed to insert data %lu %lu %i\n", src, dst, val);
        return false;
    }
    if (update_counts.count(src) == 0) {
        update_counts[src] = 0;
    }
    update_counts[src]++;
    // printf("Node %lu has count %lu\n", src, update_counts[src]);

    // free memory
    freeDBT(toInsert);
    free(value_dbt.data);

    if (update_counts[src] == TAU) { // exactly equal to avoid adding a bunch of times
        while (GraphWorker::queue_lock.test_and_set(std::memory_order_acquire))
            ; // spin-lock on the queue
        GraphWorker::work_queue.push(src);
        GraphWorker::queue_lock.clear(std::memory_order_release); // unlock
    }
    return true;
}

std::vector<uint64_t> TokuInterface::getEdges(uint64_t node) {
    int err;
    bool endOfTree= false;
    std::vector<uint64_t> ret = std::vector<uint64_t>();

    DBC* cursor = nullptr;
    DBT* cursorValue = new DBT();
    cursorValue->flags |= DB_DBT_MALLOC;

    DBT* cursorKey = new DBT();
    DBT* startDBT = toDBT(node, 0, 0); // start is node with edge to 0
    DBT* endDBT = toDBT(node, (uint64_t) -1, (uint32_t) -1);
    memcpy(cursorKey, startDBT, sizeof(DBT));
    cursorKey->flags |= DB_DBT_MALLOC;

    if((err = db->cursor(db, nullptr, &cursor, 0)) != 0) { //set up the cursor for moving through the db
        printf("Error getting cursor. %d: %s", err, db_strerror(err));
        throw std::runtime_error("Error getting cursor.");
    }

    // move cursor to first position
    err = cursor->c_get(cursor, cursorKey, cursorValue, DB_SET_RANGE);
    if (err != 0) {
        if (err == DB_NOTFOUND) {
            throw std::runtime_error("Empty query!");
        } else {
            throw std::runtime_error("Error setting range.");
        }
        if (cursor->c_close(cursor)) {
            // TODO: error message
        }
        // free memory and exit
        freeDBT(cursorKey);
        freeDBT(cursorValue);
        freeDBT(startDBT);
        freeDBT(endDBT);
        return ret;
    }

    while (keyCompare(db, endDBT, cursorKey) >= 0) {
        // uint64_t node = be64toh(getNode(cursorKey));
        uint64_t edgeTo = be64toh(getEdgeTo(cursorKey));
        int8_t value = getValue(cursorValue);

        if (value != 0) {
            // printf("Query got data %lu -> %lu, %d\n", node, edgeTo, value);
            ret.push_back(edgeTo);
            
            // insert a delete to the root for this key
            db->del(db, nullptr, cursorKey, 0);
        }
        // free memory
        free(cursorKey->data);
        free(cursorValue->data);

        err = cursor->c_get(cursor, cursorKey, cursorValue, DB_NEXT);
        if (err != 0) {
            if (err != DB_NOTFOUND) {
                // TODO: error messages
            } else {
                // There is no more data (end of tree)
                // this is a special case in which
                // cursorKey and cursorValue will not
                // have new data within them
                // therefore return a different way
                endOfTree = true;
                break;
            }
        }
    }

    if (cursor->c_close(cursor) != 0) {
        // TODO: error messages
    }

    // free and return
    if (endOfTree) {
        delete cursorKey;
        delete cursorValue;
    } else {
        freeDBT(cursorKey);
        freeDBT(cursorValue);
    }
    freeDBT(startDBT);
    freeDBT(endDBT);

    // subtract from the map tracking the number of updates
    update_counts[node] -= ret.size();

    return ret;
}

void TokuInterface::flush() {
    printf("Flushing tokudb of any remaining updates\n");
    while (GraphWorker::queue_lock.test_and_set(std::memory_order_acquire))
        ; // spin-lock on the queue
    for (auto const& pair : update_counts) {
        if (pair.second > 0 && pair.second < TAU) { // number of updates is greater than 0 and less than tau (if >= then already in queue)
            GraphWorker::work_queue.push(pair.first);
        }
    }
    GraphWorker::queue_lock.clear(std::memory_order_release); // unlock
}
