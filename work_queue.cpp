//
// Created by victor on 6/2/21.
//

#include "include/work_queue.h"
#include "include/graph_worker.h"

const unsigned first_idx = 2;

WorkQueue::WorkQueue(uint32_t buffer_size, Node nodes, int queue_len) :
buffer_size(buffer_size), cq(queue_len,buffer_size*sizeof(Node)),
buffers(nodes) {
  for (Node i = 0; i < nodes; ++i) {
    buffers[i] = static_cast<Node *>(malloc(buffer_size * sizeof(Node)));
    buffers[i][0] = first_idx; // first spot will point to the next free space
    buffers[i][1] = i; // second spot identifies the node to which the buffer
    // belongs
  }
  /* 
  mq_attr attr;
  attr.mq_maxmsg = 4;
  attr.mq_msgsize = (buffer_size * sizeof(Node) + 100);
  mqd = mq_open ("/BufferTree", O_CREAT | O_EXCL | O_WRONLY | O_NONBLOCK,  0600, &attr);

  if (mqd == (mqd_t) -1)
    {
      std::cout << "Error: " << errno << std::endl;
      exit(EXIT_FAILURE);
    }
   */
}

WorkQueue::~WorkQueue() {
  for (auto & buffer : buffers) {
    free(buffer);
  }
}

void WorkQueue::flush(Node *buffer, uint32_t num_bytes) {
  cq.push(reinterpret_cast<char *>(buffer), num_bytes);
  //push_data();
}

insert_ret_t WorkQueue::insert(update_t upd) {
  Node& idx = buffers[upd.first][0];
  buffers[upd.first][idx] = upd.second;
  ++idx;
  if (idx == buffer_size) { // full, so request flush
    flush(buffers[upd.first], buffer_size*sizeof(Node));
    idx = first_idx;
  }
}
/*
void WorkQueue::push_data() {
    data_ret_t data;
    bool extracted_data = get_data(data);
    if (!extracted_data)
      {
        return;
      }

    std::vector<char> serialized = serialize_data_ret_t(data);
    data_ret_t new_dat = deserialize_data_ret_t(serialized.data());

    int err = mq_send(mqd, serialized.data(), serialized.size(), 10);
    while (err)
      {
        err = mq_send(mqd, serialized.data(), serialized.size(), 10);
        if (errno != EAGAIN)
          {
            std::cout << errno << std::endl;
            exit(EXIT_FAILURE);
          }
      }
      
}*/

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

