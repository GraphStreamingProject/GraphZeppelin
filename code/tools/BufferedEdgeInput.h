#pragma once

#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <thread>

// Assumes that node indices are at most 2^32

class BufferedEdgeInput
{
 public:
  BufferedEdgeInput(const std::string &path, size_t buffer_size, bool insert_delete_tag = true) :
    num_nodes(),
    num_edges(),
    mutex(),
    writeBufferSignal(),
    buffers(),
    readBuffer(&buffers[0]),
    writeBuffer(&buffers[1]),
    BUFFER_SIZE(buffer_size),
    currentElement(buffer_size),
    stream(path, ios_base::in | ios_base::binary),
    swapWaiting(false),
    endOfFile(false),
    insert_delete_tag(insert_delete_tag),
    reader()
  {
    // TODO: Handle portability issues involving endianess
    stream.read(reinterpret_cast<char *>(&num_nodes), 4);
    stream.read(reinterpret_cast<char *>(&num_edges), 8);
    reader = std::thread(&BufferedEdgeInput::ingest, this);
    readBuffer->reserve(buffer_size);
    writeBuffer->reserve(buffer_size);
  }

  ~BufferedEdgeInput()
  {
    reader.join();
  }

  bool get_edge(std::tuple<uint32_t, uint32_t, uint8_t> &out)
  {
    if (currentElement >= readBuffer->size())
    {
      readBuffer->clear();
      {
        std::unique_lock<std::mutex> lk(mutex);
        writeBufferSignal.wait(lk, [&](){return swapWaiting || endOfFile;});
        std::swap(writeBuffer, readBuffer);
        currentElement = 0;
        swapWaiting = false;
      }
      writeBufferSignal.notify_all();
    }
    
    if (currentElement < readBuffer->size())
    {
      out = (*readBuffer)[currentElement++];
      return true;
    }
    
    return false;
  }

  void ingest()
  {
    while (true)
    {
      for (size_t i = 0; i < BUFFER_SIZE; ++i)
      {
        uint32_t src_vertex;
	uint32_t dst_vertex;
	uint8_t tag;

        if (!(stream.read(reinterpret_cast<char *>(&tag), 1)))
        {
	  std::unique_lock<std::mutex> lk(mutex);
	  swapWaiting = true;
	  endOfFile = true;
	  writeBufferSignal.notify_all();
	  return;
        }

	// TODO: Handle portability issues with endianess

        stream.read(reinterpret_cast<char *>(&src_vertex), 4);
	stream.read(reinterpret_cast<char *>(&dst_vertex), 4);

        writeBuffer->emplace_back(src_vertex, dst_vertex, tag);
      }

      {
        std::unique_lock<std::mutex> lk(mutex);
	swapWaiting = true;
	writeBufferSignal.notify_all();
        writeBufferSignal.wait(lk, [&](){return !swapWaiting;});
      }
    }
  }

  uint32_t num_nodes;
  uint64_t num_edges;
  
 private:
  std::mutex mutex;
  std::condition_variable writeBufferSignal;
  std::vector<std::tuple<uint32_t, uint32_t, uint8_t>> buffers[2];
  std::vector<std::tuple<uint32_t, uint32_t, uint8_t>>* readBuffer;
  std::vector<std::tuple<uint32_t, uint32_t, uint8_t>>* writeBuffer;
  const size_t BUFFER_SIZE;
  size_t currentElement;
  ifstream stream;
  bool swapWaiting;
  bool endOfFile;
  bool insert_delete_tag;
  std::thread reader;
};
