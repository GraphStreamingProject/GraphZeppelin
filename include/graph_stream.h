#pragma once
#include <exception>
#include <string>
#include <unordered_map>

#include "types.h"

#pragma pack(push,1)
struct GraphStreamUpdate {
  uint8_t type;
  Edge edge;
};
#pragma pack(pop)

static constexpr edge_id_t END_OF_STREAM = (edge_id_t) -1;

// Enum that defines the types of streams
enum StreamType {
  BinaryFile,
  AsciiFile,
};

class GraphStream {
 public:
  virtual ~GraphStream() = default;
  inline node_id_t vertices() { return num_vertices; }
  inline edge_id_t edges() { return num_edges; }

  // Extract a buffer of many updates from the stream
  virtual size_t get_update_buffer(GraphStreamUpdate* upd_buf, edge_id_t num_updates) = 0;

  // Query the GraphStream to see if get_update_buffer is thread-safe
  // this is implemenation dependent
  virtual bool get_update_is_thread_safe() = 0;

  // Move read pointer to new location in stream
  // Child classes may choose to throw an error if seek is called
  // For example, a GraphStream recieved over the network would
  // likely not support seek
  virtual void seek(edge_id_t edge_idx) = 0;

  // Query handling
  // Call this function to register a query at a future edge index
  // This function returns true if the query is correctly registered
  virtual bool set_break_point(edge_id_t query_idx) = 0;

  // Serialize GraphStream metadata for distribution
  // So that stream reading can happen simultaneously
  virtual void serialize_metadata(std::ostream &out) = 0;

  // construct a stream object from serialized metadata
  static GraphStream* construct_stream_from_metadata(std::istream &in);

 protected:
  node_id_t num_vertices = 0;
  edge_id_t num_edges = 0;
 private:
  static std::unordered_map<size_t, GraphStream* (*)(std::istream&)> constructor_map;
};

class StreamException : public std::exception {
 private:
  std::string err_msg;
 public:
  StreamException(std::string err) : err_msg(err) {}
  virtual const char* what() const throw() { return err_msg.c_str(); }
};