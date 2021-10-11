#pragma once
#include <buffer_tree.h>
// TODO: switch references from Node to node_t in types

/**
 * In-memory wrapper to offer the same interface as a buffer tree.
 */
class WorkQueue {
  const uint32_t buffer_size; // size of a buffer (including metadata)
  CircularQueue cq;
  std::vector<std::vector<node_id_t>> buffers; // array dump of numbers for performance: DO NOT
  // try to access directly!

  /**
   * Flushes the corresponding buffer to the queue.
   * @param buffer      a pointer to the head of the buffer to flush.
   * @param num_bytes   the number of bytes to flush.
   */
  void flush(std::vector<node_id_t> &buffer, uint32_t num_bytes);
public:
  /**
   * Constructs a new queue.
   * @param size        the total length of a buffer, in updates.
   * @param nodes       number of nodes in the graph.
   * @param queue_len   the length of the work queue.
   */
  WorkQueue(uint32_t size, Node nodes, int queue_len);

  ~WorkQueue();

  /**
   * Puts an update into the data structure.
   * @param upd the edge update.
   * @return nothing.
   */
  insert_ret_t insert(update_t upd);

  /**
   * Ask the buffer queue for data and sleep if necessary until it is available.
   * @param data        to store the fetched data.
   * @return            true if got valid data, false if unable to get data.
   */
  bool get_data(data_ret_t& data);

  /**
   * Flushes all pending buffers.
   * @return nothing.
   */
  flush_ret_t force_flush();

  /**
   * Notifies all threads waiting on condition variables that they should
   * check their wait condition again. Useful when switching from blocking to
   * non-blocking calls to the circular queue.
   * For example: we set this to true when shutting down the graph_workers.
   * @param    block is true if we should turn on non-blocking operations
   *           and false if we should turn them off.
   */
  void set_non_block(bool block);
};
