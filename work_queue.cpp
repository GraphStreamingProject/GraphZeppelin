//
// Created by victor on 6/2/21.
//

#include "include/work_queue.h"
#include "include/graph_worker.h"

const unsigned first_idx = 2;

WorkQueue::WorkQueue(uint32_t buffer_size, Node nodes, int queue_len) :
buffer_size(buffer_size), cq(queue_len,buffer_size*sizeof(Node)),
buffers() {
  for (Node i = 0; i < nodes; ++i) {
    buffers.emplace_back();
    buffers[i].push_back(first_idx);
    buffers[i].push_back(i);
  }
}

WorkQueue::~WorkQueue() {
}

void WorkQueue::flush(Node *buffer, uint32_t num_bytes) {
  cq.push(reinterpret_cast<char *>(buffer), num_bytes);
}

insert_ret_t WorkQueue::insert(const update_t &upd) {
  std::vector<Node> &ptr = buffers[upd.first];
  ptr.emplace_back(upd.second);
  if (ptr.size() == buffer_size) { // full, so request flush
    ptr[0] = buffer_size;
    flush(ptr.data(), buffer_size*sizeof(Node));
    Node i = ptr[1];
    ptr.clear();
    ptr.push_back(first_idx);
    ptr.push_back(i);
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

  for (uint32_t j = first_idx; j < vec_len; ++j) {
    data.second.push_back(serial_data[j]);
  }

  cq.pop(i); // mark the cq entry as clean
  return true;
}

flush_ret_t WorkQueue::force_flush() {
  for (auto & buffer : buffers) {
    if (buffer.size() > first_idx) { // have stuff to flush
      buffer[0] = buffer.size();
      flush(buffer.data(), buffer[0]*sizeof(Node));
      Node i = buffer[1];
      buffer.clear();
      buffer.push_back(0);
      buffer.push_back(i);
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

data_ret_t WorkQueue::deserialize_data_ret_t(const char* buf)
{
  Node first = *reinterpret_cast<const Node*>(buf);
  size_t length = *reinterpret_cast<const size_t*>(buf + sizeof(Node));
  std::vector<Node> second;
  for (size_t i = 0; i < length; ++i)
    {
      const auto offset = buf + sizeof(Node) + sizeof(size_t);
      Node node = *reinterpret_cast<const Node*>(offset + i * sizeof(Node));
      second.push_back(node);
    }
  return std::make_pair<Node, std::vector<Node>>(std::move(first), std::move(second));
}


std::vector<char> WorkQueue::serialize_data_ret_t(const data_ret_t &data)
{
  std::vector<char> serialized_data;
  for (size_t i = 0; i < sizeof(data.first); ++i)
    {
      serialized_data.push_back(reinterpret_cast<const char*>(&data.first)[i]);
    }
  size_t size = data.second.size();
  for (size_t i = 0; i < sizeof(size); ++i)
    {
      serialized_data.push_back(reinterpret_cast<const char*>(&size)[i]);
    }
  for (size_t i = 0; i < data.second.size(); ++i)
    {
      const Node *node = &data.second[i];
      for (size_t j = 0; j < sizeof(Node); ++j)
        {
          serialized_data.push_back(reinterpret_cast<const char*>(node)[j]);
        }
    }
  return serialized_data;
}

