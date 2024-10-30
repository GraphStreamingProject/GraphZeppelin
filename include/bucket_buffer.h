#include "sketch.h"
#include <algorithm>
#include <cassert>


/**
 * A coo-matrix-style representation of a sketch. 
 * Note that we sort them by the row index (depth) and then by column indices.
 */
struct BufferEntry {
    int col_idx;
    int row_idx;
    Bucket value;
    bool operator==(const BufferEntry &rhs) const {
        return std::tie(row_idx, col_idx) == std::tie(rhs.row_idx, rhs.col_idx);
    }
    bool operator!=(const BufferEntry &rhs) const {
        return !(*this == rhs);
    }
    bool operator<(const BufferEntry &rhs) const {
        return std::tie(row_idx, col_idx) < std::tie(rhs.row_idx, rhs.col_idx);
    }
    bool operator>(const BufferEntry &rhs) const {
        return rhs < *this;
    }
    bool operator<=(const BufferEntry &rhs) const {
        return !(*this > rhs);
    }
    bool operator>=(const BufferEntry &rhs) const {
        return !(*this < rhs);
    }
};


class BucketBuffer {
    friend class Sketch;
    public:
    BufferEntry *entries;

    private:
    size_t _capacity;
    size_t _size = 0;

    bool _compacted=false;

    // should be about the largest expected buffer size.
    // TODO - change this design to be more flexible?
    static constexpr size_t BUFFER_CAPACITY = 256;
    BufferEntry thread_local_buffer[BUFFER_CAPACITY];
    BufferEntry thread_local_otherbuffer[BUFFER_CAPACITY];

    public:
    BucketBuffer() : entries(new BufferEntry[32]), _capacity(32) {}
    BucketBuffer(
        BufferEntry *entries, size_t _capacity) :
        entries(entries),
         _capacity(_capacity) {}
    ~BucketBuffer() {
        // delete[] entries;
    }
    
    size_t size() const {
        return _size;
    }

    bool insert(int col_idx, int row_idx, Bucket value) {
        // note: if this ever returns false, it's time for you to grow your 
        // sketch size.
        if (_size >= _capacity) {
            sort_and_compact();
            if (_size >= _capacity) {
                return false;
            }
        }
        entries[_size++] = {col_idx, row_idx, value};
        return true;
    }

    const BufferEntry &operator[](size_t idx) const {
        return entries[idx];
    }

    void sort_and_compact() {
        return;
        // The goal of this operation is primarily to combine entries in the
        // buffer that have the same col_idx and row_idx. 
        if (_compacted) {
            return;
        }
        std::sort(entries, entries + _size, std::greater<BufferEntry>());

        size_t write_idx = 0;
        for (size_t read_idx = 1; read_idx < _size; ++read_idx) {
            // skip entries that are empty
            if (entries[read_idx].col_idx == entries[write_idx].col_idx &&
                entries[read_idx].row_idx == entries[write_idx].row_idx) {
                entries[write_idx].value ^= entries[read_idx].value;
            } else {
                entries[++write_idx] = entries[read_idx];
            }
        }
        _size = write_idx + 1;

        // get rid of entries with a value of 0
        write_idx = 0;
        for (size_t read_idx = 0; read_idx < _size; ++read_idx) {
            if (!Bucket_Boruvka::is_empty(entries[read_idx].value)) {
                entries[write_idx++] = entries[read_idx];
            }
        }
        _size = write_idx;

        _compacted = true;
    }


    // merge another buffer into this one
    // returns false if the merge would exceed the capacity of the buffer, does not perform the merge
    // returns true otherwise
    bool merge(const BucketBuffer &other) {
        // easy case - can just copy one buffer into the other
        likely_if (_size + other._size <= _capacity)  {
            std::copy(other.entries, other.entries + other._size, entries + _size);
            _size += other._size;
            return true;
        }
        // otherwise, see if we can merge the two buffers
        assert(_size + other._size <= BUFFER_CAPACITY);
        size_t buffer_size = 0;

        std::copy(other.entries, other.entries + other._size, thread_local_otherbuffer);
        // sort both yourself and the other buffer
        if (!_compacted)
            std::sort(entries, entries + _size, std::greater<BufferEntry>());
        if (!other._compacted) 
            std::sort(
                thread_local_otherbuffer, 
                thread_local_otherbuffer + other._size,
                std::greater<BufferEntry>()
            );

        // standard sorted-merge on the two buffers
        // except we XOR the values if the col_idx and row_idx are the same
        // NOTE WE WANT TO SORT DESCENDING
        size_t i = 0, j = 0;
        BufferEntry to_append = {0, 0, {0, 0}};        
        while (i < _size && j < other._size) {
            // ascending sort!!!
            if (entries[i] >= thread_local_otherbuffer[j]) {
                to_append = entries[i++];
            } else {
                to_append = thread_local_otherbuffer[j++];
            }
            if (buffer_size == 0 || to_append != thread_local_buffer[buffer_size - 1]) {
                thread_local_buffer[buffer_size++] = to_append;
            } else {
                thread_local_buffer[buffer_size - 1].value ^= to_append.value;
            }
        }

        // copy over the remaining entries with the compaction scheme
        while (i < _size) {
            if (buffer_size == 0 || entries[i] != thread_local_buffer[buffer_size - 1]) {
                thread_local_buffer[buffer_size++] = entries[i++];
            } else {
                thread_local_buffer[buffer_size - 1].value ^= entries[i++].value;
            }
        }
        while (j < other._size) {
            if (buffer_size == 0 || thread_local_otherbuffer[j] != thread_local_buffer[buffer_size - 1]) {
                thread_local_buffer[buffer_size++] = thread_local_otherbuffer[j++];
            } else {
                thread_local_buffer[buffer_size - 1].value ^= thread_local_otherbuffer[j++].value;
            }
        }

        //now remove entries that are 0'd out
        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < buffer_size; ++read_idx) {
            if (!Bucket_Boruvka::is_empty(thread_local_buffer[read_idx].value)) {
                thread_local_buffer[write_idx++] = thread_local_buffer[read_idx];
            }
        }
        buffer_size = write_idx;

        unlikely_if (buffer_size > _capacity) {
            return false;
        }
        else {
            std::copy(thread_local_buffer, thread_local_buffer + buffer_size, entries);
            _size = buffer_size;
            _compacted = true;
            return true;
        }
    }

    void clear() {
        _size = 0;
        _compacted = false;
    }
};

