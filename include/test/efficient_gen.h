#pragma once

void write_edges(int n, double p, std::string out_f);
// insert, delete based on a geometric distribution with ratio p
// i.e. p% of edges will be deleted, p^2% will be re-inserted, p^3 will be re-deleted
// until 1 element is left
void insert_delete(double p, std::string in_file, std::string out_file);

void write_cumul(const std::string& stream_f, const std::string& cumul_f);
