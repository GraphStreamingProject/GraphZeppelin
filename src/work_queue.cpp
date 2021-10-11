#include <cassert>
#include "../include/work_queue.h"
#include "../include/graph_worker.h"
#include "../include/types.h"

WorkQueue::WorkQueue(uint32_t buffer_size, node_t nodes, int queue_len) :
buffer_size(buffer_size), cq(queue_len,buffer_size*sizeof(node_id_t)),
buffers(nodes) {
  for (Node i = 0; i < nodes; ++i) {
    buffers.emplace_back();
    buffers[i].reserve(buffer_size);
    buffers[i].push_back(i);
  }
}

WorkQueue::~WorkQueue() {
}

void WorkQueue::flush(std::vector<node_id_t> &buffer, uint32_t num_bytes) {
  cq.push(reinterpret_cast<char *>(buffer.data()), num_bytes);
}

insert_ret_t WorkQueue::insert(update_t upd) {
  auto &buf = buffers[upd.first];
  buf.push_back(upd.second);
  if (buf.size() == buffer_size) { // full, so request flush
    flush(buf, buffer_size*sizeof(node_id_t));
    buf.clear();
    buf.push_back(upd.first);
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
  auto *serial_data = reinterpret_cast<node_id_t *>(elm.data);
  uint32_t len      = elm.size;
  assert(len % sizeof(node_id_t) == 0);

  if (len == 0)
    return false; // we got no data so return not valid

  // assume the first key is correct so extract it
  node_t key = serial_data[0];
  data.first = key;

  data.second.clear(); // remove any old data from the vector
  uint32_t vec_len  = len / sizeof(node_id_t);
  data.second.reserve(vec_len); // reserve space for our updates

  for (uint32_t j = 1; j < vec_len; ++j) {
    data.second.push_back(serial_data[j]);
  }

  cq.pop(i); // mark the cq entry as clean
  return true;
}

flush_ret_t WorkQueue::force_flush() {
  for (auto & buffer : buffers) {
    if (!buffer.empty()) { // have stuff to flush
      flush(buffer, buffer.size()*sizeof(node_id_t));
      buffer.clear();
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
