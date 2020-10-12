// main.cpp
#include <xxhash.h>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
using namespace std;

int main()
{
    size_t const bufferSize = 10;
    void* const buffer = malloc(bufferSize);
    XXH64_hash_t hash = XXH64(buffer, bufferSize, time(NULL));
    XXH64_hash_t hash2 = XXH64(buffer, bufferSize, 0);
    printf("%llu\n",hash);
    cout << hash2 << endl;
    return 0;
}
