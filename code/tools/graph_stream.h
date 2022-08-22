#include <fstream>
#include <cstring>

class GraphStream {
public:
	GraphStream(const char *file_name, uint32_t b) : buf_size(b) {
		file_stream.open(file_name);
		buf = (char *) malloc(buf_size * sizeof(char));
		start_buf = buf;

		// read the header from the file
		file_stream >> nodes >> edges;

		// read in first buffer from file
		file_stream.read(buf, buf_size);
		eof = !file_stream.good();
	}
	~GraphStream() {
		free(start_buf);
		file_stream.close();
	}
	inline void parse_line(uint8_t *u, uint32_t *a, uint32_t *b) {
		if (buf_size - (buf - start_buf) < 64 && !eof)
			eof = read_data();

		uint32_t temp_a = 0, temp_b = 0;

		// skip newline if present
		if (*buf < '0' || *buf > '9') buf++;
		
		// first character is u
		*u = *buf - '0';
		buf++;
		// skip tab
		buf++;

		// now parse a
		while (*buf >= '0' && *buf <= '9') {
			temp_a = (temp_a * 10) + (*buf - '0');
			buf++;
		}
		*a = temp_a;
		// skip tab
		buf++;

		// now parse b
		while (*buf >= '0' && *buf <= '9') {
			temp_b = (temp_b * 10) + (*buf - '0');
			buf++;
		}
		*b = temp_b;
	}

	uint32_t nodes;
	uint64_t edges;

private:
	uint32_t buf_size;
	bool eof = false;
	char *buf;
	char *start_buf;
	std::ifstream file_stream;

	inline bool read_data() {
		int rem = buf_size - (buf - start_buf);
		std::memcpy(start_buf, buf, rem);
		buf = start_buf + rem;
		
		// read the data
		file_stream.read(buf, buf_size - rem);
		
		// now set buf to beginning
		buf = start_buf;
		return !file_stream.good();
	}
};

