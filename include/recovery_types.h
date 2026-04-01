#pragma once
#include <vector>
#include "types.h"

enum RecoveryResultTypes {
    // success in retrieving everything
    SUCCESS,
    // failure because there are too many
    // things in the sketch to recover
    FAILURE,
    // we are decently sure that this
    // would have succeeded if we had a small number
    // of extra cleanup sketches.
    PARTIAL_RECOVERY
};

struct RecoveryResult {
    RecoveryResultTypes result;
    std::vector<vec_t> recovered_indices;
};
