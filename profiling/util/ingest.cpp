#include "ingest.h"

vector<set<Node>> ingest_con_comp (istream * updates_stream)
{
	int n, m;
        *updates_stream >> n >> m;

        Graph G{n};

        int t, u, v;
        UpdateType type;
        for (int i = 0; i < m; i++)
        {
                *updates_stream >> t >> u >> v;
                type = t ? DELETE : INSERT;

                G.update({{u, v}, type});
        }

        return G.connected_components();
}

vector<unordered_map<Node, vector<Node>>> 
ingest_span_forest (istream * updates_stream)
{
	int n, m;
        *updates_stream >> n >> m;

        Graph G{n};

        int t, u, v;
        UpdateType type;
        for (int i = 0; i < m; i++)
        {
                *updates_stream >> t >> u >> v;
                type = t ? DELETE : INSERT;

                G.update({{u, v}, type});
        }

        return G.spanning_forest();
}

vector<vector<Node>> 
ingest_k_edge_disjoint_span_forests_union (istream * updates_stream)
{
	int n, m;
        *updates_stream >> n >> m;

        Graph G{n};

        int t, u, v;
        UpdateType type;
        for (int i = 0; i < m; i++)
        {
                *updates_stream >> t >> u >> v;
                type = t ? DELETE : INSERT;

                G.update({{u, v}, type});
        }

        return G.k_edge_disjoint_span_forests_union();
}
