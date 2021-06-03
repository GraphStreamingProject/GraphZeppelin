//
// Created by victor on 6/2/21.
//

#include "include/work_queue.h"

const unsigned first_idx = 2;

WorkQueue::WorkQueue(uint32_t size, Node nodes) : size(size), cq(20, size*sizeof
(Node)),
buffers(nodes) {
  for (Node i = 0; i < nodes; ++i) {
    buffers[i] = static_cast<unsigned long *>(malloc(size * sizeof(Node)));
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

void WorkQueue::flush(Node *buffer, uint32_t length) {
  cq.push(reinterpret_cast<char *>(buffer), length);
}

insert_ret_t WorkQueue::insert(update_t upd) {
  Node& idx = buffers[upd.first][0];
  buffers[upd.first][idx] = upd.second;
  ++idx;
  if (idx == size) { // full, so request flush
    flush(buffers[upd.first], size*sizeof(Node));
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
  Node *serial_data = reinterpret_cast<Node *>(elm.data);
  uint32_t len      = elm.size;
  assert(len % sizeof(Node) == 0);

  if (len == 0)
    return false; // we got no data so return not valid

  // assume the first key is correct so extract it
  Node key = serial_data[1];
  data.first = key;

  data.second.clear(); // remove any old data from the vector
  uint32_t vec_len  = len / sizeof(Node);
  data.second.reserve(vec_len); // reserve space for our updates

  for (uint32_t j = first_idx; j < len / sizeof(Node); ++j) {
    data.second.push_back(serial_data[j]);
  }

  cq.pop(i); // mark the cq entry as clean
  return true;
}

flush_ret_t WorkQueue::force_flush() {
  for (auto & buffer : buffers) {
    if (buffer[0] != first_idx) { // have stuff to flush
      flush(buffer, buffer[0]);
      buffer[0] = first_idx;
    }
  }
}
