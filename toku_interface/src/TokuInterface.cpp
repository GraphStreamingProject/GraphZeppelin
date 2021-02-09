#include "TokuInterface.h"

#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <exception>
#include <stdexcept>
#include <chrono>

#define USE_DEFAULT false
#define NEW_DB_DIR "../../graph-db-data"

inline int keyCompare(DB* db __attribute__((__unused__)), const DBT *a, const DBT *b) {
    return memcmp(a->data, b->data, a->size);
}

inline DBT *toDBT(uint64_t src, uint64_t dst) {
    DBT *key_dbt = new DBT();
    key_dbt->flags=0;
    key_dbt->size = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint64_t);
    key_dbt->ulen = key_dbt->size;
    key_dbt->data = malloc(key_dbt->size);
    memcpy(key_dbt->data, &src, sizeof(uint64_t));
    memcpy((uint8_t *)key_dbt->data + sizeof(uint64_t), &dst, sizeof(uint64_t));
    uint64_t uid = std::chrono::system_clock::now().time_since_epoch().count();
    memcpy((uint8_t *)key_dbt->data + 2*sizeof(uint64_t), &uid, sizeof(uint64_t));
    return key_dbt;
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
    char* dbfile = (char*) calloc(MAX_DB_FILENAME_LENGTH, sizeof(char));
    char* dbdir = (char*) calloc(MAX_ENV_DIRNAME_LENGTH, sizeof(char));// = {'\0'};

    strncpy(dbfile, DB_FILE, MAX_DB_FILENAME_LENGTH);
    dbfile[MAX_DB_FILENAME_LENGTH - 1] = '\0';

    strncpy(dbdir, DB_DIR, MAX_ENV_DIRNAME_LENGTH);
    dbdir[MAX_ENV_DIRNAME_LENGTH - 1] = '\0';


    int dbFlags = DB_CREATE;
    int envFlags = DB_PRIVATE|DB_INIT_MPOOL|DB_INIT_LOCK|DB_THREAD|DB_CREATE;

    db_env_set_direct_io(false);

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

    // TODO: set cachesize
    // TODO: set redzone

    if (env->open(env, dbdir, envFlags, S_IRWXU|S_IRWXG) != 0) {
        printf("Failed to open env!\n");
        exit(EXIT_FAILURE);
    }

    printf("Successfully opened env\n");

    if (db_create(&db, env, 0) != 0) {
        printf("error creating db.");
        exit(EXIT_FAILURE);
    }

    // TODO: set buffersize
    // TODO: set fanout

    if (db->open(db, NULL, dbfile, NULL, DB_BTREE, dbFlags, S_IRUSR|S_IWUSR|S_IRGRP) != 0) {
        printf("Error opening db!\n");
        exit(EXIT_FAILURE);
    }
    free(dbfile);
    free(dbdir);

    // create map which will store the number of unqueries updates
    // to the datastructure
    update_counts = std::unordered_map<uint64_t, uint64_t>();
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

bool TokuInterface::putEdge(std::pair<uint64_t, uint64_t> edge, int8_t value) {
    if (!putSingleEdge(edge.first, edge.second, value))
        return false;

    return putSingleEdge(edge.second, edge.first, value);
}

bool TokuInterface::putSingleEdge(uint64_t src, uint64_t dst, int8_t val) {
    DBT value_dbt;
    value_dbt.flags=0;
    value_dbt.size = sizeof(int8_t);
    value_dbt.ulen = sizeof(int8_t);
    value_dbt.data = malloc(sizeof(int8_t));
    memset(value_dbt.data, val, sizeof(int8_t));
    if (db->put(db, NULL, toDBT(src, dst), &value_dbt, 0) != 0) {
        // TODO error message
        return false;
    }
    return true;
}


std::vector<std::pair<uint64_t, int8_t>>* TokuInterface::getEdges(uint64_t node) {
    int err;
    std::vector<std::pair<uint64_t, int8_t>> *ret = new std::vector<std::pair<uint64_t, int8_t>>();

    DBC* cursor = nullptr;
    DBT* cursorValue = new DBT();
    cursorValue->flags |= DB_DBT_MALLOC;

    DBT* cursorKey = new DBT();
    DBT* startDBT = toDBT(node, 0); // start is node with edge to 0
    memcpy(cursorKey, startDBT, sizeof(DBT));
    cursorKey->flags |= DB_DBT_MALLOC;

    err = db->cursor(db, nullptr, &cursor, 0); //set up the cursor for moving through the db

    if (err != 0) {
        if (err == DB_NOTFOUND) {
            throw std::runtime_error("Empty query!");
        } else {
            throw std::runtime_error("Error setting range.");
        }
        if (cursor->c_close(cursor)) {
            // TODO: error message
        }
        return ret;
    }

    while (keyCompare(db, toDBT(node, (uint64_t) -1), cursorKey) >= 0) {
        // uint64_t node = getNode(cursorKey);
        uint64_t edgeTo = getEdgeTo(cursorKey);
        int8_t value = getValue(cursorValue);

        if (value != 0) {
            ret->push_back(std::pair<uint64_t, int8_t>(edgeTo, value));
            
            // insert a delete to the root for this key
            db->del(db, nullptr, cursorKey, 0);

            // free memory
            free(cursorKey->data);
            free(cursorValue->data);
        }


        err = cursor->c_get(cursor, cursorKey, cursorValue, DB_NEXT);
        if (err != 0) {
            if (err != DB_NOTFOUND) {
                // TODO: error messages
            } else {
                // Done with reading the data
                break;
            }
        }
    }

    if (cursor->c_close(cursor) != 0) {
        // TODO: error messages
    }

    delete cursorKey;
    delete cursorValue;
    return ret;
}

