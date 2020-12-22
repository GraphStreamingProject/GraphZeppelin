#ifndef TEST_INT127_H
#define TEST_INT127_H

#include <exception>
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint128_t;
using boost::multiprecision::int128_t;

/**
 * Space-efficient 127-bit storage integer based on Boost's uint128_t type.
 */
class int127 {
  uint128_t m_value;
public:
  int127(const int127& other);
  int127(const int128_t& other);
  int127(const unsigned long& other);
  int127(const long& other);
  int127(const int& other);

  void operator=(const int127& other);
  void operator=(const int128_t& other);
  void operator=(const unsigned long& other);
  void operator=(const long& other);
  void operator=(const int& other);

  friend int127 operator+(const int127& lhs, const int127& rhs);
  friend int127 operator-(const int127& lhs, const int127& rhs);
  friend int127& operator+=(int127& lhs, const int127& rhs);
  friend int127 operator/(const int127& lhs, const int127& rhs);
  friend int127 operator%(const int127& lhs, const int127& rhs);
  friend bool operator==(const int127& lhs, const int127& rhs);
  friend bool operator!=(const int127& lhs, const int127& rhs);
  friend bool operator>(const int127& lhs, const int127& rhs);
  friend bool operator<=(const int127& lhs, const int127& rhs);
  friend std::ostream & operator<<(std::ostream & os, const int127& value );

  friend int127 operator*(const int127& lhs, const long& rhs);
  friend bool operator!=(const int127& lhs, const long long int& rhs);

  int128_t toBoostInt128() const;
  uint128_t toBoostUInt128() const;
};

class Int127OverflowException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "Computation would overflow int127 type!";
  }
};

class InvalidUInt128CastException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "Cannot cast a negative to unsigned type!";
  }
};

#endif //TEST_INT127_H
