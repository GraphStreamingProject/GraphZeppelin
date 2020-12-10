#pragma once

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
 * @throws overflow_error if there would be an overflow in computing the function.
 */
uint64_t nondirectional_non_self_edge_pairing_fn(uint64_t i, uint64_t j);

/**
 * Inverts the nondirectional non-SE pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
std::pair<uint64_t , uint64_t> inv_nondir_non_self_edge_pairing_fn(uint64_t idx);

/**
 * Implementation of the Cantor diagonal pairing function.
 * @throws overflow_error if there would be an overflow in computing the function.
 */
uint64_t cantor_pairing_fn(uint64_t i, uint64_t j);