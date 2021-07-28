//
// Created by victor on 6/2/21.
//

#include "include/work_queue.h"
#include "include/graph_worker.h"

const unsigned first_idx = 2;

WorkQueue::WorkQueue(uint32_t buffer_size, Node nodes, int queue_len) :
buffer_size(buffer_size), cq(queue_len,buffer_size*sizeof(node_id_t)),
buffers(nodes) {
  for (Node i = 0; i < nodes; ++i) {
    buffers[i] = static_cast<node_id_t *>(malloc(buffer_size * sizeof(node_id_t)));
    buffers[i][0] = first_idx; // first spot will point to the next free space
    buffers[i][1] = i; // second spot identifies the node to which the buffer
    // belongs
  }
}

WorkQueue::~WorkQueue() {
  for (auto & buffer : buffers) {
    free(buffer);
  }
}

void WorkQueue::flush(Node *buffer, uint32_t num_bytes) {
  cq.push(reinterpret_cast<char *>(buffer), num_bytes);
}

insert_ret_t WorkQueue::insert(update_t upd) {
  Node& idx = buffers[upd.first][0];
  buffers[upd.first][idx] = upd.second;
  ++idx;
  if (idx == buffer_size) { // full, so request flush
    flush(buffers[upd.first], buffer_size*sizeof(node_id_t));
    idx = first_idx;
  }
}

// basically a copy of BufferTree::get_data()
bool WorkQueue::get_data(data_ret_t &data) {
  // make a request to the circular buffer for data
  std::pair<int, queue_elm> queue_data;
  bool got_data = cq.peek(queue_data);

  if (!got_data)
    return false; // we got no data so return not valid

  int i         = queue_data.first;
  queue_elm elm = queue_data.second;
  Node *serial_data = reinterpret_cast<node_id_t *>(elm.data);
  uint32_t len      = elm.size;
  assert(len % sizeof(node_id_t) == 0);

  if (len == 0)
    return false; // we got no data so return not valid

  // assume the first key is correct so extract it
  Node key = serial_data[1];
  data.first = key;

  data.second.clear(); // remove any old data from the vector
  uint32_t vec_len  = len / sizeof(node_id_t);
  data.second.reserve(vec_len); // reserve space for our updates

  for (uint32_t j = first_idx; j < vec_len; ++j) {
    data.second.push_back(serial_data[j]);
  }

  cq.pop(i); // mark the cq entry as clean
  return true;
}

flush_ret_t WorkQueue::force_flush() {
  for (auto & buffer : buffers) {
    if (buffer[0] != first_idx) { // have stuff to flush
      flush(buffer, buffer[0]*sizeof(Node));
      buffer[0] = first_idx;
    }
  }
}

void WorkQueue::set_non_block(bool block) {
  if (block) {
    cq.no_block = true; // circular queue operations should no longer block
    cq.cirq_empty.notify_all();
  } else {
    cq.no_block = false; // set circular queue to block if necessary
  }
}
