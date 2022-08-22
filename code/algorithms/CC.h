// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of 
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// #include "ligra.h"
#include "math.h"
#pragma once
#define newA(__E,__n) (__E*) malloc((__n)*sizeof(__E))

template <class ET>
inline bool CAS(ET *ptr, ET oldv, ET newv) {
  if (sizeof(ET) == 1) {
    return __sync_bool_compare_and_swap((bool*)ptr, *((bool*)&oldv), *((bool*)&newv));
  } else if (sizeof(ET) == 4) {
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&oldv), *((int*)&newv));
  } else if (sizeof(ET) == 8) {
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&oldv), *((long*)&newv));
  }
  else {
    std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
    abort();
  }
}

template <class ET>
inline bool writeMin(ET *a, ET b) {
    ET c; bool r=0;
    do c = *a;
    while (c > b && !(r=CAS(a,c,b)));
    return r;
}

struct CC_Shortcut {
  uintE* IDs, *prevIDs;
  bool *output;
  CC_Shortcut(uintE* _IDs, uintE* _prevIDs, bool* _output) :
    IDs(_IDs), prevIDs(_prevIDs), output(_output) {}
  inline bool operator () (uintE i) {
    uintE l = IDs[IDs[i]];
    if(IDs[i] != l) IDs[i] = l;
    if(prevIDs[i] != IDs[i]) {
      prevIDs[i] = IDs[i];
      output[i] = 1;
      return 1; }
    else return 0;
  }
};

//function used by vertex map to sync prevIDs with IDs
struct CC_Vertex_F {
  uintE* IDs, *prevIDs;
  CC_Vertex_F(uintE* _IDs, uintE* _prevIDs) :
    IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool operator () (uintE i) {
    prevIDs[i] = IDs[i];
    return 1; }};

/* for with shortcut
struct CC_F {
  uintE* IDs, *prevIDs;
  CC_F(uintE* _IDs, uintE* _prevIDs) : 
    IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool update(uintE s, uintE d){ //Update function writes min ID
    uintE origID = IDs[d];
    if(IDs[s] < origID) {
      IDs[d] = min(origID,IDs[s]);
    } return 1; }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update
    uintE origID = IDs[d];
    writeMin(&IDs[d],IDs[s]);
    return 1;
  }
  inline bool cond (uintE d) { return true; } //does nothing
};
*/

struct CC_F {
  uintE* IDs, *prevIDs;
  CC_F(uintE* _IDs, uintE* _prevIDs) : 
    IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool update(uintE s, uintE d){ //Update function writes min ID
    uintE origID = IDs[d];
    if(IDs[s] < origID) {
      IDs[d] = min(origID,IDs[s]);
      if(origID == prevIDs[d]) return 1;
    } return 0; }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update
    uintE origID = IDs[d];
    return (writeMin(&IDs[d],IDs[s]) && origID == prevIDs[d]);
  }
  inline bool cond (uintE d) { return true; } //does nothing
};
template <class Graph>
void CC(Graph& G /*, commandLine& P */) {
  size_t n = G.num_vertices();
	auto vtxs = G.fetch_all_vertices();
	uintE* IDs = newA(uintE,n);
	uintE* prevIDs = newA(uintE,n);

  parallel_for(0, n, [&] (size_t i) {  IDs[i] = i; } );
  //parallel_for(0, n, [&] (size_t i) {  prevIDs[i] = i; } );
  
	//bool* all = newA(bool,n);
  //parallel_for(0, n, [&] (size_t i) {  all[i] = 1; } );

  // dense vertex subset
  //vertex_subset All(n,n,all);

	bool* active = newA(bool,n);
	parallel_for(0, n, [&] (size_t i) {  active[i] = 1; } );
  vertex_subset Active(n,n,active);
  
  timer sparse_t, dense_t, fetch_t, other_t;

  while(!Active.is_empty()) {
		vertex_map(Active, CC_Vertex_F(IDs,prevIDs));
		vertex_subset output = G.edge_map(Active, CC_F(IDs, prevIDs), vtxs, sparse_t, dense_t, other_t);
		Active.del();
		Active = output;
  }
	Active.del();
	free(prevIDs);
  /*
  ofstream ofile;
  ofile.open("ids.txt");
  for(uint32_t i = 0; i < n; i++) {
    ofile << IDs[i] << "\n";
  }
  ofile.close();
  */
  free(IDs);
}

