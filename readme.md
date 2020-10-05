Tasks for implementing l_0 sampling:
1. replace the trivial hash function with xxhash.
2. rewrite code to determine whether an index belongs in a bucket
3. add in c code: note that p must be prime and larger than poly(n) = n^2
4. write code to guess mu (/ phi) instead of assuming it
5. make the code modular
