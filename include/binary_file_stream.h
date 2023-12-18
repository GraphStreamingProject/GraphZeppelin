#pragma once
#include <fcntl.h>
#include <unistd.h>  //open and close

#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

#include "graph_stream.h"

class BinaryFileStream : public GraphStream {
 public:
  /**
   * Open a BinaryFileStream
   * @param file_name   Name of the stream file
   */
  BinaryFileStream(std::string file_name, bool open_read_only = true)
      : read_only(open_read_only), file_name(file_name) {
    if (read_only)
      stream_fd = open(file_name.c_str(), O_RDONLY, S_IRUSR);
    else
      stream_fd = open(file_name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);

    if (!stream_fd)
      throw StreamException("BinaryFileStream: Could not open stream file " + file_name +
                            ". Does it exist?");

    // read header from the input file
    if (read_only) {
      if (read(stream_fd, (char*)&num_vertices, sizeof(num_vertices)) != sizeof(num_vertices))
        throw StreamException("BinaryFileStream: Could not read number of nodes");
      if (read(stream_fd, (char*)&num_edges, sizeof(num_edges)) != sizeof(num_edges))
        throw StreamException("BinaryFileStream: Could not read number of edges");

      end_of_file = (num_edges * edge_size) + header_size;
      stream_off = header_size;
      set_break_point(-1);
    }
  }

  ~BinaryFileStream() {
    if (stream_fd) close(stream_fd);
  }

  inline size_t get_update_buffer(GraphStreamUpdate* upd_buf, size_t num_updates) {
    assert(upd_buf != nullptr);

    // many threads may execute this line simultaneously creating edge cases
    size_t bytes_to_read = num_updates * edge_size;
    size_t read_off = stream_off.fetch_add(bytes_to_read, std::memory_order_relaxed);

    // catch these edge cases here
    if (read_off + bytes_to_read > break_index) {
      bytes_to_read = read_off > break_index ? 0 : break_index - read_off;
      stream_off = break_index.load();
      upd_buf[bytes_to_read / edge_size] = {BREAKPOINT, {0, 0}};
    }
    // read into the buffer
    assert(bytes_to_read % edge_size == 0);
    size_t bytes_read = 0;
    while (bytes_read < bytes_to_read) {
      int r =
          pread(stream_fd, upd_buf + bytes_read, bytes_to_read - bytes_read, read_off + bytes_read);
      if (r == -1) throw StreamException("BinaryFileStream: Could not perform pread");
      if (r == 0) throw StreamException("BinaryFileStream: pread() got no data");
      bytes_read += r;
    }

    size_t upds_read = bytes_to_read / edge_size;
    if (upds_read < num_updates) {
      GraphStreamUpdate& upd = upd_buf[upds_read];
      upd.type = BREAKPOINT;
      upd.edge = {0, 0};
      return upds_read + 1;
    }
    return upds_read;
  }

  // get_update_buffer() is thread safe! :)
  inline bool get_update_is_thread_safe() { return true; }

  // write the number of nodes and edges to the stream
  inline void write_header(node_id_t num_verts, edge_id_t num_edg) {
    if (read_only) throw StreamException("BinaryFileStream: stream not open for writing!");

    lseek(stream_fd, 0, SEEK_SET);
    int r1 = write(stream_fd, (char*)&num_verts, sizeof(num_verts));
    int r2 = write(stream_fd, (char*)&num_edg, sizeof(num_edg));

    if (r1 + r2 != header_size) {
      perror("write_header");
      throw StreamException("BinaryFileStream: could not write header to stream file");
    }

    stream_off = header_size;
    num_vertices = num_verts;
    num_edges = num_edg;
    end_of_file = (num_edges * edge_size) + header_size;
  }

  // write an edge to the stream
  inline void write_updates(GraphStreamUpdate* upd, edge_id_t num_updates) {
    if (read_only) throw StreamException("BinaryFileStream: stream not open for writing!");

    size_t bytes_to_write = num_updates * edge_size;
    // size_t write_off = stream_off.fetch_add(bytes_to_write, std::memory_order_relaxed);

    size_t bytes_written = 0;
    while (bytes_written < bytes_to_write) {
      int r = write(stream_fd, (char*)upd + bytes_written, bytes_to_write - bytes_written);
      if (r == -1) throw StreamException("BinaryFileStream: Could not perform write");
      bytes_written += r;
    }
  }

  // seek to a position in the stream
  inline void seek(edge_id_t edge_idx) { stream_off = edge_idx * edge_size + header_size; }

  inline bool set_break_point(edge_id_t break_idx) {
    edge_id_t byte_index = END_OF_STREAM;
    if (break_idx != END_OF_STREAM) {
      byte_index = header_size + break_idx * edge_size;
    }
    if (byte_index < stream_off) return false;
    break_index = byte_index;
    if (break_index > end_of_file) break_index = end_of_file;
    return true;
  }

  inline void serialize_metadata(std::ostream& out) {
    out << BinaryFile << " " << file_name << std::endl;
  }

  static GraphStream* construct_from_metadata(std::istream& in) {
    std::string file_name_from_stream;
    in >> file_name_from_stream;
    return new BinaryFileStream(file_name_from_stream);
  }

 private:
  int stream_fd;
  edge_id_t end_of_file;
  std::atomic<edge_id_t> stream_off;
  std::atomic<edge_id_t> break_index;
  const bool read_only;  // is stream read only?
  const std::string file_name;

  // size of binary encoded edge and buffer read size
  static constexpr size_t edge_size = sizeof(GraphStreamUpdate);
  static constexpr size_t header_size = sizeof(node_id_t) + sizeof(edge_id_t);
};
