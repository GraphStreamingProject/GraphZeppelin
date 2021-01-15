#include "../include/int127.h"

using namespace boost::multiprecision::literals;
constexpr uint128_t high_bitmask = 0x80000000000000000000000000000000_cppui128;
constexpr int128_t low_bitmask =   0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF_cppi128;

int128_t int127::toBoostInt128() const {
  if (m_value & high_bitmask) {
    int128_t retval(m_value);
    return -(retval&low_bitmask);
  } else return {m_value};
}


uint128_t int127::toBoostUInt128() const {
  if (m_value & high_bitmask)
    throw InvalidUInt128CastException();
  return m_value;
}

int127::int127(const int127& other) {
  m_value = other.m_value;
}

int127::int127(const int128_t &other) {
  if (abs(other) & high_bitmask)
    throw Int127OverflowException();
  if (other < 0) {
    m_value = static_cast<uint128_t> ((-other) + high_bitmask);
  } else m_value = static_cast<uint128_t>(other);
}

int127::int127(const unsigned long &other) {
  m_value = other;
}

int127::int127(const long &other) {
  if (other < 0) {
    m_value = high_bitmask + (-other);
  } else m_value = other;
}

int127::int127(const int &other) {
  if (other < 0) {
    m_value = high_bitmask + (-other);
  } else m_value = other;
}

int127& int127::operator=(const int127 &other) = default;

int127& int127::operator=(const boost::multiprecision::int128_t &other) {
  if (abs(other) & high_bitmask)
    throw Int127OverflowException();
  if (other < 0) {
    m_value = static_cast<uint128_t> ((-other) + high_bitmask);
  } else m_value = static_cast<uint128_t>(other);
  return *this;
}

int127& int127::operator=(const unsigned long &other) {
  m_value = other;
  return *this;
}

int127& int127::operator=(const long &other) {
  if (other < 0) {
    m_value = high_bitmask + (-other);
  } else m_value = other;
  return *this;
}

int127& int127::operator=(const int &other) {
  if (other < 0) {
    m_value = high_bitmask + (-other);
  } else m_value = other;
  return *this;
}

int127 operator+(const int127 &lhs, const int127 &rhs) {
  return {lhs.toBoostInt128() + rhs.toBoostInt128()};
}

int127 operator-(const int127 &lhs, const int127 &rhs) {
  return {lhs.toBoostInt128() - rhs.toBoostInt128()};
}

int127 &operator+=(int127 &lhs, const int127 &rhs) {
  int128_t temp = lhs.toBoostInt128() + rhs.toBoostInt128();
  if (abs(temp) & high_bitmask)
    throw Int127OverflowException();
  if (temp < 0) {
    lhs.m_value = static_cast<uint128_t> ((-temp) + high_bitmask);
  } else lhs.m_value = static_cast<uint128_t>(temp);
  return lhs;
}

int127 operator/(const int127 &lhs, const int127 &rhs) {
  return {lhs.toBoostInt128() / rhs.toBoostInt128()};
}

int127 operator%(const int127& lhs, const int127& rhs) {
  return {lhs.toBoostInt128() % rhs.toBoostInt128()};
}

bool operator==(const int127& lhs, const int127& rhs) {
  return lhs.m_value == rhs.m_value;
}

bool operator!=(const int127& lhs, const int127& rhs) {
  return lhs.m_value != rhs.m_value;
}

bool operator>(const int127 &lhs, const int127 &rhs) {
  return lhs.toBoostInt128() > rhs.toBoostInt128();
}

bool operator<=(const int127 &lhs, const int127 &rhs) {
  return lhs.toBoostInt128() <= rhs.toBoostInt128();
}

std::ostream &operator<<(std::ostream &os, const int127 &value) {
  if (value.m_value & high_bitmask) os << "-";
  os << (value.m_value & low_bitmask);
  return os;
}

int127 operator*(const int127 &lhs, const long &rhs) {
  return {lhs.toBoostInt128()*rhs};
}

bool operator!=(const int127& lhs, const long long int& rhs) {
  return lhs.m_value != rhs;
}
