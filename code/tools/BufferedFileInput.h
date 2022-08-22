#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <thread>

class BufferedFileInput
{
 public:
  BufferedFileInput(const std::string &path, size_t buffer_size) :
    mutex(),
    writeBufferSignal(),
    buffers(),
    readBuffer(&buffers[0]),
    writeBuffer(&buffers[1]),
    BUFFER_SIZE(buffer_size),
    currentElement(buffer_size),
    stream(path),
    reader(&BufferedFileInput::ingest, this),
    swapWaiting(false),
    endOfFile(false)
  {
    readBuffer->reserve(buffer_size);
    writeBuffer->reserve(buffer_size);
  }

  ~BufferedFileInput()
  {
    reader.join();
  }

  bool getline(std::string &out)
  {
    if (currentElement >= readBuffer->size())
    {
      {
        std::unique_lock<std::mutex> lk(mutex);
        writeBufferSignal.wait(lk, [&](){return swapWaiting || endOfFile;});
	readBuffer->clear();
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
        std::string line;
        if (!std::getline(stream, line))
        {
	  std::unique_lock<std::mutex> lk(mutex);
	  swapWaiting = true;
	  endOfFile = true;
	  writeBufferSignal.notify_all();
	  return;
        }

        writeBuffer->push_back(line);
      }

      {
        std::unique_lock<std::mutex> lk(mutex);
	swapWaiting = true;
	writeBufferSignal.notify_all();
        writeBufferSignal.wait(lk, [&](){return !swapWaiting;});
      }
    }
  }
  
 private:
  std::mutex mutex;
  std::condition_variable writeBufferSignal;
  std::vector<std::string> buffers[2];
  std::vector<std::string>* readBuffer;
  std::vector<std::string>* writeBuffer;
  const size_t BUFFER_SIZE;
  size_t currentElement;
  ifstream stream;
  std::thread reader;
  bool swapWaiting;
  bool endOfFile;
};
