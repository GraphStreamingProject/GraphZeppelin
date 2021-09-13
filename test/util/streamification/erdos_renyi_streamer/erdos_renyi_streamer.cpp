#include <chrono>
#include <random>

template <class node_index>
class erdos_renyi_streamer
{
    public:
	erdos_renyi_streamer (size_t n, unsigned seed = std::chrono::system_clock::now().time_since_epoch().count())
	{
	    gen = std::mt19937{seed};
	    first_node = std::uniform_int_distribution<node_index>(0, n - 1);
	    second_node = std::uniform_int_distribution<node_index>(0, n - 2);
	}

	std::pair<node_index, node_index> next()
	{
	    node_index first = first_node(gen);
	    node_index second = second_node(gen);
	    
	    // Avoid self-loops
	    if (second >= first)
		    second++;

	    return std::make_pair(first, second);
	}
	
    private:
	std::mt19937 gen;
	std::uniform_int_distribution<node_index> first_node;
	std::uniform_int_distribution<node_index> second_node;
};

