#include "sketch.h"
#include <algorithm>
#include <cassert>
#include <array>

/**
 * A coo-matrix-style representation of a sketch. 
 * Note that we sort them by the row index (depth) and then by column indices.
 */
struct BufferEntry {
    Bucket value;
    int col_idx;
    int row_idx;
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

// class BucketBufferHashMap {
//     public:
//     // std::unordered_map<int, std::unordered_map<int, Bucket>> entries;
//     std::unordered_map<std::pair<int, int>, Bucket> entries;
//     size_t _capacity;
//     BucketBufferHashMap(size_t capacity): _capacity(capacity) {};

//     bool insert(int col_idx, int row_idx, Bucket value) {
//         if (entries.size() >= _capacity) {
//             return false;
//         }
//         static constexpr Bucket zero_bucket = {0, 0};
//         entries.emplace(std::make_pair(col_idx, row_idx), zero_bucket);
//         entries[{col_idx, row_idx}] ^= value;
//         if (Bucket_Boruvka::is_empty(entries[{col_idx, row_idx}])) {
//             entries.erase({col_idx, row_idx});
//         }
//         return true;
//     }
//     bool merge(const BucketBufferHashMap &other) {
//         for (const auto &idx : other.entries) {
//             static constexpr Bucket zero_bucket = {0, 0};
//             entries.emplace(idx.first, zero_bucket);
//             entries[idx.first] ^= idx.second;
//             if (Bucket_Boruvka::is_empty(entries[idx.first])) {
//                 entries.erase(idx.first);
//             }
//         }
//         // TODO - make this less gross
//         unlikely_if (entries.size() >= _capacity) {
//             // UNDO THE MERGE
//             for (const auto &idx : other.entries) {
//                 static constexpr Bucket zero_bucket = {0, 0};
//                 entries.emplace(idx.first, zero_bucket);
//                 entries[idx.first] ^= idx.second;
//                 if (Bucket_Boruvka::is_empty(entries[idx.first])) {
//                     entries.erase(idx.first);
//                 }
//             }
//             return false;
//         }
//         else
//             return true;
//     }
// };


// note that we consider these to be 
class BucketBuffer {
    friend class Sketch;
    public:
    std::vector<BufferEntry> entries;

    private:
    size_t _capacity;

    bool _compacted=false;

    // should be about the largest expected buffer size.
    // TODO - change this design to be more flexible?
    // static constexpr size_t BUFFER_CAPACITY = 256;
    // std::array<BufferEntry, BUFFER_CAPACITY> thread_local_buffer {};
    // std::array<BufferEntry, BUFFER_CAPACITY> thread_local_otherbuffer {};

    public:
    BucketBuffer(): _capacity(20) {
        entries = std::vector<BufferEntry>();
        entries.reserve(_capacity);
    }
    ~BucketBuffer() {
    }

    bool over_capacity() const {
        return entries.size() >= _capacity / 2;
    }
    
    size_t size() const {
        return entries.size();
    }

    /*
        * Insert a value into the buffer. 
        * If the buffer is full, it will compact itself.
        * If the buffer is still full after compaction, it will return false.
    */
    bool insert(int col_idx, int row_idx, Bucket value) {
        _compacted = false;
        entries.emplace_back(BufferEntry({value, col_idx, row_idx}));
        if (over_capacity()) {
            sort_and_compact();
            return !over_capacity();
        }
        return true;
    }

    const BufferEntry &operator[](size_t idx) const {
        return entries[idx];
    }

    void sort_and_compact() {
        // The goal of this operation is primarily to combine entries in the
        // buffer that have the same col_idx and row_idx. 
        if (_compacted) {
            return;
        }
        std::sort(entries.begin(), entries.end(), std::greater<BufferEntry>());
        size_t _size = entries.size();

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
        entries.resize(_size);

        _compacted = true;
        // if (size() > 5) {
        //     for (size_t i = 0; i < _size; ++i) {
        //         std::cout << "(" << entries[i].col_idx << ", " << entries[i].row_idx << ") ";
        //     }
        //     std::cout << std::endl;
        // }
    }

    bool merge(const BucketBuffer &other) {
        // YOU SHOULD ONLY MERGE WITH AN UNDER CAPACITY BUFFER
        // TODO - for now, that is the responsibility of the caller.
        // std::cout << "Merging buffers of size " << size() << " and " << other.size() << std::endl;
        assert(size() + other.size() <= _capacity);
        entries.insert(entries.end(), other.entries.begin(), other.entries.end());
        _compacted = false;
        // TODO - use a more efficient sorting procedure
        if (over_capacity()) {
            sort_and_compact();
        }
        return !over_capacity();
    }

    // merge another buffer into this one
    // returns false if the merge would exceed the capacity of the buffer, does not perform the merge
    // returns true otherwise
    // bool merge(const BucketBuffer &other) {
    //TODO - go look at commit if you want to restore this version
        // assert(size() + other.size() <= _capacity);
        // entries.insert(entries.end(), other.entries.begin(), other.entries.end());
        // // otherwise, see if we can merge the two buffers
        // assert(size() + other.size() <= BUFFER_CAPACITY);
        // size_t buffer_size = 0;

        // // std::copy(other.entries, other.entries + other._size, thread_local_otherbuffer);
        // std::copy(other.entries.begin(), other.entries.end(), thread_local_otherbuffer.begin());
        // // sort both yourself and the other buffer
        // if (!_compacted)
        //     std::sort(entries.begin(), entries.end(), std::greater<BufferEntry>());
        // if (!other._compacted) 
        //     std::sort(
        //         thread_local_otherbuffer.begin(), 
        //         thread_local_otherbuffer.begin() + other.size(),
        //         std::greater<BufferEntry>()
        //     );

        // // standard sorted-merge on the two buffers
        // // except we XOR the values if the col_idx and row_idx are the same
        // // NOTE WE WANT TO SORT DESCENDING
        // size_t i = 0, j = 0;
        // BufferEntry to_append = {{0, 0}, 0, 0};        
        // while (i < size() && j < other.size()) {
        //     // ascending sort!!!
        //     if (entries[i] >= thread_local_otherbuffer[j]) {
        //         to_append = entries[i++];
        //     } else {
        //         to_append = thread_local_otherbuffer[j++];
        //     }
        //     if (buffer_size == 0 || to_append != thread_local_buffer[buffer_size - 1]) {
        //         thread_local_buffer[buffer_size++] = to_append;
        //     } else {
        //         thread_local_buffer[buffer_size - 1].value ^= to_append.value;
        //     }
        // }

        // // copy over the remaining entries with the compaction scheme
        // while (i < size()) {
        //     if (buffer_size == 0 || entries[i] != thread_local_buffer[buffer_size - 1]) {
        //         thread_local_buffer[buffer_size++] = entries[i++];
        //     } else {
        //         thread_local_buffer[buffer_size - 1].value ^= entries[i++].value;
        //     }
        // }
        // while (j < other.size()) {
        //     if (buffer_size == 0 || thread_local_otherbuffer[j] != thread_local_buffer[buffer_size - 1]) {
        //         thread_local_buffer[buffer_size++] = thread_local_otherbuffer[j++];
        //     } else {
        //         thread_local_buffer[buffer_size - 1].value ^= thread_local_otherbuffer[j++].value;
        //     }
        // }

        // //now remove entries that are 0'd out
        // size_t write_idx = 0;
        // for (size_t read_idx = 0; read_idx < buffer_size; ++read_idx) {
        //     if (!Bucket_Boruvka::is_empty(thread_local_buffer[read_idx].value)) {
        //         thread_local_buffer[write_idx++] = thread_local_buffer[read_idx];
        //     }
        // }
        // buffer_size = write_idx;

        // unlikely_if (buffer_size > _capacity) {
        //     return false;
        // }
        // else {
        //     std::copy(thread_local_buffer.begin(), thread_local_buffer.begin() + buffer_size, entries.begin());
        //     entries.resize(buffer_size);
        //     _compacted = true;
        //     return true;
        // }
    // }

    void clear() {
        entries.resize(0);
        _compacted = false;
    }
};

