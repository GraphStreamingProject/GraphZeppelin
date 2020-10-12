Tasks for implementing l_0 sampling:
1. replace the trivial hash function with xxhash.
2. rewrite code to determine whether an index belongs in a bucket
3. add in c code: note that p must be prime and larger than poly(n) = n^2
4. write code to guess mu (/ phi) instead of assuming it
5. make the code modular

longer term tasks:
- making different sketches combinable
- making sure each sketch (log^2 collection of buckets) is only ever sampled from ONCE
- testing output correctness and distribution uniformity. also testing memory usage
- some amount of optimization

Notes for future optimization:
- Miller-Rabin vs Fermat vs AKS (for primality testing)? Currently using AKS
- xxHash is non-trivial to install. Should maybe figure out a way to streamline/automate installation process
- The c variable for each bucket seems inefficient to compute, since you need to compute r^i, where i = O(N)
- Can the algorithm ever return a false positive (i.e. an index whose entry is actually zero)? If so, is the probability of this affected by the guessing of phi (the number of nonzero entries?

10/12/20:
- Find existing implementation of modular exponentiation
- Two ways the algorithm can fail: if the buckets are all bad
- Is c-test affected by guessing of mu? If so, should we check buckets in a particular order?

- Make script to automate installation of xxhash
- Modify generation of random number r to use the xxhash seed
