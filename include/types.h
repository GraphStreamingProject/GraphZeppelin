#pragma once

#ifdef USE_NATIVE_F
//Native type includes
#  ifndef __SIZEOF_INT128__
#include <boost/multiprecision/cpp_int.hpp>
#  endif
#else //USE_NATIVE_F
//128 bit includes
#include <boost/multiprecision/cpp_int.hpp>
//TODO: low priority: when using larger types in 32 bit system, use int127
//#  ifndef __SIZEOF_INT128__
//#  include "int127.h"
//#  endif
#endif //USE_NATIVE_F

#ifdef __SIZEOF_INT128__
//overload operator<< for __int128 and __uint128
#include <iostream>
#endif //__SIZEOF_INT_128__

#ifdef USE_NATIVE_F
//Native types
typedef uint32_t node_t; //Max graph vertices is 92682
typedef uint32_t vec_t; //Max sketch vector size is 2^32 - 1
typedef int64_t bucket_t;
typedef uint64_t ubucket_t;
#  ifdef __SIZEOF_INT128__
typedef __int128 bucket_prod_t;
typedef unsigned __int128 ubucket_prod_t;
#  else //__SIZEOF_INT_128__
typedef boost::multiprecision::int128_t bucket_prod_t;
typedef boost::multiprecision::uint128_t ubucket_prod_t;
#  endif
#else //USE_NATIVE_F
//128 bit types
typedef uint64_t node_t; //Max graph vertices is 6074001000
typedef uint64_t vec_t; //Max sketch vector size is 2^64 - 1
#  ifdef __SIZEOF_INT128__
typedef __int128 bucket_t;
typedef unsigned __int128 ubucket_t;
#  else //__SIZEOF_INT_128__
typedef boost::multiprecision::int128_t bucket_t;
typedef boost::multiprecision::uint128_t ubucket_t;
#  endif //__SIZEOF_INT_128__
typedef boost::multiprecision::int256_t bucket_prod_t;
typedef boost::multiprecision::uint256_t ubucket_prod_t;
#endif //USE_NATIVE_F

#ifdef __SIZEOF_INT128__
//overload operator<< for __int128 and __uint128
//Low priority TODO: Detect order in which __int128 is stored?
//Kenny: Although my machine is little endian, it seems like the high quadword
//is stored after the low quadword.
inline std::ostream& operator<<(std::ostream& os, const __int128& n) {
	return os << std::hex << reinterpret_cast<const uint64_t*>(&n)[1] << reinterpret_cast<const uint64_t*>(&n)[0];
}
inline std::ostream& operator<<(std::ostream& os, const unsigned __int128& n) {
	return os << std::hex << reinterpret_cast<const uint64_t*>(&n)[1] << reinterpret_cast<const uint64_t*>(&n)[0];
}
#else //__SIZEOF_INT_128__
//larger types in 32 bit system, add definition for log2
//Add definition for log2
inline double log2(const boost::multiprecision::int128_t& x) {
	return log2(x.convert_to<double>());
}
#endif //__SIZEOF_INT_128__
