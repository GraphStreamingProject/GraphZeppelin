#pragma once

#ifdef USE_NATIVE_F
//Native type includes
#ifndef __SIZEOF_INT128__
#include <boost/multiprecision/cpp_int.hpp>
#endif
#else
//128 bit includes
#include <boost/multiprecision/cpp_int.hpp>
//TODO: low priority: when using larger types in 32 bit system, use int127
//#ifndef __SIZEOF_INT128__
//#include "int127.h"
//#endif
#endif

#ifdef __SIZEOF_INT128__
//overload operator<< for __int128 and __uint128
#include <iostream>
#endif

#ifdef USE_NATIVE_F
//Native types
typedef uint32_t node_t; //Max graph vertices is 92682
typedef uint32_t vec_t; //Max sketch vector size is 2^32 - 1
typedef int64_t bucket_t;
typedef uint64_t ubucket_t;
#ifdef __SIZEOF_INT128__
typedef __int128 bucket_prod_t;
typedef unsigned __int128 ubucket_prod_t;
#else
typedef boost::multiprecision::int128_t bucket_prod_t;
typedef boost::multiprecision::uint128_t ubucket_prod_t;
#endif
#else
//128 bit types
typedef uint64_t node_t; //Max graph vertices is 92682
typedef uint64_t vec_t; //Max sketch vector size is 2^64 - 1
#ifdef __SIZEOF_INT128__
typedef __int128 bucket_t;
typedef unsigned __int128 ubucket_t;
#else
typedef boost::multiprecision::int128_t bucket_t;
typedef boost::multiprecision::uint128_t ubucket_t;
#endif
typedef boost::multiprecision::int256_t bucket_prod_t;
typedef boost::multiprecision::uint256_t ubucket_prod_t;
#endif

#ifdef __SIZEOF_INT128__
//overload operator<< for __int128 and __uint128
inline std::ostream& operator<<(std::ostream& os, const __int128& n) {
	//TODO: Endianness?
	return os << std::hex << reinterpret_cast<const uint64_t*>(&n)[1] << reinterpret_cast<const uint64_t*>(&n)[0];
}

inline std::ostream& operator<<(std::ostream& os, const unsigned __int128& n) {
	//TODO: Endianness?
	return os << std::hex << reinterpret_cast<const uint64_t*>(&n)[1] << reinterpret_cast<const uint64_t*>(&n)[0];
}
#else
//larger types in 32 bit system, add definition for log2
//Add definition for log2
inline double log2(const boost::multiprecision::int128_t& x) {
	return log2(x.convert_to<double>());
}
#endif
