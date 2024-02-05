#pragma once

#include <fstream>
#include <iostream>
#include <cassert>

#include "graph_stream.h"

class AsciiFileStream : public GraphStream {
 public:
  AsciiFileStream(std::string file_name, bool has_type = true)
      : file_name(file_name), has_type(has_type) {

    bool stream_exists = false;
    {
      std::fstream check(file_name, std::fstream::in);
      stream_exists = check.is_open();
    }

    if (stream_exists)
      stream_file.open(file_name, std::fstream::in | std::fstream::out);
    else
      stream_file.open(file_name, std::fstream::in | std::fstream::out | std::fstream::trunc);

    if (!stream_file.is_open())
      throw StreamException("AsciiFileStream: could not open " + file_name);

    if (stream_exists)
      stream_file >> num_vertices >> num_edges;
  }

  inline size_t get_update_buffer(GraphStreamUpdate* upd_buf, size_t num_updates) {
    assert(upd_buf != nullptr);

    size_t i = 0;
    for (; i < num_updates; i++) {
      GraphStreamUpdate& upd = upd_buf[i];

      if (upd_offset >= num_edges || upd_offset >= break_edge_idx) {
        upd.type = BREAKPOINT;
        upd.edge = {0, 0};
        return i + 1;
      }
      int type = INSERT;
      if (has_type)
        stream_file >> type;
      stream_file >> upd.edge.src >> upd.edge.dst;
      upd.type = type;
      ++upd_offset;
    }
    return i;
  }

  // get_update_buffer() is not thread safe
  inline bool get_update_is_thread_safe() { return false; }

  inline void write_header(node_id_t num_verts, edge_id_t num_edg) {
    stream_file.seekp(0); // seek to beginning
    stream_file << num_verts << " " << num_edg << std::endl;
    num_vertices = num_verts;
    num_edges = num_edg;
  }

  inline void write_updates(GraphStreamUpdate* upd_buf, edge_id_t num_updates) {
    for (edge_id_t i = 0; i < num_updates; i++) {
      auto upd = upd_buf[i];
      if (has_type)
        stream_file << (int) upd.type << " ";
      stream_file << upd.edge.src << " " << upd.edge.dst << std::endl;
    }
  }

  inline void set_num_edges(edge_id_t num_edg) {
    num_edges = num_edg;
  }

  inline void seek(edge_id_t pos) {
    if (pos != 0)
      throw StreamException("AsciiFileStream: stream does not support seeking by update index");
    stream_file.seekp(0); stream_file.seekg(0);
    upd_offset = 0;
  }

  inline bool set_break_point(edge_id_t break_idx) {
    if (break_idx < upd_offset) return false;
    break_edge_idx = break_idx;
    return true;
  }

  inline void serialize_metadata(std::ostream& out) {
    out << AsciiFile << " " << file_name << std::endl;
  }

  static GraphStream* construct_from_metadata(std::istream& in) {
    std::string file_name_from_stream;
    in >> file_name_from_stream;
    return new AsciiFileStream(file_name_from_stream);
  }

 private:
  const std::string file_name;
  const bool has_type;
  std::fstream stream_file;
  edge_id_t break_edge_idx = -1;
  edge_id_t upd_offset = 0;
};
