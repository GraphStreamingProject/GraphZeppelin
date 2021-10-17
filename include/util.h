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

/**
 * Configures the system using the configuration file streaming.conf
 * Gets the path prefix where the buffer tree data will be stored and sets
 * with the number of threads used for a variety of tasks.
 * Should be called before creating the buffer tree or starting graph workers.
 * @return the prefix of the path in which the buffer tree should be stored.
 */
std::string configure_system();

/**
 * Utility function to convert a cumulative graph file to an adjacency matrix
 * representation.
 * Accepts graphs of the form
 *      n m
 *      a_1 b_1
 *      a_2 b_2
 *      ...
 *      a_m b_m
 * representing a graph with n (0-indexed) nodes and m edges, where the i-th
 * edge is between nodes a_1 and b_1.
 * @param file
 * @return (upper-triangular) adjacency matrix.
 */
std::vector<bool> cum_file_to_adj_matrix(const std::string& file);
