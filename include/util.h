#pragma once
#include <tuple>

/**
 * Cast a double to unsigned long long with epsilon adjustment.
 * @param d         the double to cast.
 * @param epsilon   optional parameter representing the epsilon to use.
 */
unsigned long long int double_to_ull(double d, double epsilon = 0.00000001);

/**
 * Concatenates two node ids to form an edge ids
 * @return i + j*(j-1)/2
 */
uint64_t nondirectional_non_self_edge_pairing_fn(uint32_t i, uint32_t j);

/**
 * Inverts the nondirectional non-SE pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
std::pair<uint32_t , uint32_t> inv_nondir_non_self_edge_pairing_fn(uint64_t idx);

/**
 * Configures the system using the configuration file streaming.conf
 * Gets the path prefix where the buffer tree data will be stored and sets
 * with the number of threads used for a variety of tasks.
 * Should be called before creating the buffer tree or starting graph workers.
 * @return a tuple with the following elements
 *   - use_guttertree
 *   - in_memory_backups
 *   - disk_dir
 */
std::tuple<bool, bool, std::string> configure_system();
