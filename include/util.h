#pragma once
#include <string>
#include <tuple>

#include "types.h"

/**
 * Cast a double to unsigned long long with epsilon adjustment.
 * @param d         the double to cast.
 * @param epsilon   optional parameter representing the epsilon to use.
 */
unsigned long long int double_to_ull(double d, double epsilon = 0.00000001);

/**
 * A function N x N -> N that implements a non-self-edge pairing function
 * that does not respect order of inputs.
 * That is, (2,2) would not be a valid input. (1,3) and (3,1) would be treated as
 * identical inputs.
 * @return i + j*(j-1)/2
 */
edge_id_t nondirectional_non_self_edge_pairing_fn(node_id_t i, node_id_t j);

/**
 * Inverts the nondirectional non-SE pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
Edge inv_nondir_non_self_edge_pairing_fn(edge_id_t idx);

/**
 * Concatenates two node ids to form an edge ids
 * @return (i << 32) & j
 */
edge_id_t concat_pairing_fn(node_id_t i, node_id_t j);

/**
 * Inverts the concat pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
Edge inv_concat_pairing_fn(edge_id_t idx);

#define likely_if(x) if(__builtin_expect((bool)(x), true))
#define unlikely_if(x) if (__builtin_expect((bool)(x), false))
