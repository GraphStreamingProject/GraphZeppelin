#pragma once
#include <iostream>
#include <math.h>

#ifdef _M_IX86

#define umulrem(z, x, y, m) \
__asm mov   eax, x  \
__asm mul   y \
__asm div   m \
__asm mov   z, edx

#define umuladdrem(z, x, y, a, m) \
__asm mov   eax, x  \
__asm mul   y \
__asm add   eax, a  \
__asm adc   edx, 0  \
__asm div   m \
__asm mov   z, edx

#else

#ifdef _MSC_VER
typedef unsigned __int64  Tu64;
#else
typedef unsigned long long  Tu64;
#endif

#define umulrem(z, x, y, m) \
  { \
  z = (unsigned int)(x * (Tu64)y % m);  \
  }

#define umuladdrem(z, x, y, a, m) \
  { \
  z = (unsigned int)((x * (Tu64)y + a) % m);  \
  }

#endif

namespace AKS{
  static bool IsPrime(unsigned int n)
  {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;

    const unsigned int iMax = (int)sqrt(n) + 1;
    unsigned int i;
    for (i = 3; i <= iMax; i += 2)
      if (n % i == 0)
        return false;

    return true;
  }

  static unsigned int LargestPrimeFactor(unsigned int n)
  {
    if (n < 2) return 1;

    unsigned int r = n, p;
    if (r % 2 == 0)
    {
      p = 2;
      do { r /= 2; } while (r % 2 == 0);
    }
    unsigned int i;
    for (i = 3; i <= r; i += 2)
    {
      if (r % i == 0)
      {
        p = i;
        do { r /= i; } while (r % i == 0);
      }
    }
    return p;
  }

  /**
   * Function to calculate n to the power e, mod m.
   */
  static unsigned int Powm(unsigned int n, unsigned int e, unsigned int m)
  {
    unsigned int r = 1;
    unsigned int t = n % m;
    unsigned int i;
    for (i = e; i != 0; i /= 2)
    {
      if (i % 2 != 0)
      {
        umulrem(r, r, t, m);
      }
      umulrem(t, t, t, m);
    }
    return r;
  }

  /**
   * Function to calculate the inverse of n, mod m
   */
  static unsigned int Inv(unsigned int n, unsigned int m)
  {
    unsigned int a = n, b = m;
    int u = 1, v = 0;
    do
    {
      const unsigned int q = a / b;

      const unsigned int t1 = a - q*b;
      a = b;
      b = t1;

      const int t2 = u - (int)q*v;
      u = v;
      v = t2;
    } while (b != 0);
    if (a != 1) u = 0;
    if (u < 0) u += m;
    return u;
  }

  class CPolyMod
  {
  protected:
    // (mod x^r - 1, n)
    const unsigned int m_r;
    const unsigned int m_n;
    unsigned int m_deg;
    unsigned int * mp_a;

  private:
    CPolyMod():m_r(0), m_n(0) { mp_a = NULL; };

  public:
    // default value is x
    CPolyMod(unsigned int r, unsigned int n)
      : m_r(r), m_n(n)
    {
      m_deg = 1;
      mp_a = new unsigned int [2];
      mp_a[0] = 0; mp_a[1] = 1;
    }

    CPolyMod(const CPolyMod & p)
      : m_r(p.m_r), m_n(p.m_n)
    {
      m_deg = p.m_deg;
      mp_a = new unsigned int [p.m_deg + 1];
      unsigned int i;
      for (i = 0; i <= p.m_deg; ++i)
        mp_a[i] = p.mp_a[i];
    }

    virtual ~CPolyMod()
    {
      if (mp_a != NULL)
        delete [] mp_a;
    }

  private:
    void _polySquare()
    {
      const unsigned int deg = m_deg;
      const unsigned int n = m_n;
      const unsigned int * const p_a = mp_a;

      const unsigned int degr = deg + deg;
      unsigned int * const p_ar = new unsigned int [degr + 1];
      unsigned int k;
      for (k = 0; k <= degr; ++k)
        p_ar[k] = 0;

      unsigned int j;
      for (j = 1; j <= deg; ++j)
      {
        const unsigned int x = p_a[j];
        if (x != 0)
        {
          unsigned int i;
          for (i = 0; i < j; ++i)
          {
            const unsigned int y = 2 * p_a[i];
            unsigned int t = p_ar[j + i];
            umuladdrem(t, x, y, t, n);
            p_ar[j + i] = t;
          }
        }
      }
      unsigned int i;
      for (i = 0; i <= deg; ++i)
      {
        const unsigned int x = p_a[i];
        unsigned int t = p_ar[2 * i];
        umuladdrem(t, x, x, t, n);
        p_ar[2 * i] = t;
      }

      m_deg = degr;
      delete [] mp_a;
      mp_a = p_ar;
    }

    void _polyMul(const CPolyMod & p)
    {
      const unsigned int deg = m_deg;
      const unsigned int n = m_n;
      const unsigned int * const p_a = mp_a;

      const unsigned int degr = deg + p.m_deg;
      unsigned int * const p_ar = new unsigned int [degr + 1];
      unsigned int k;
      for (k = 0; k <= degr; ++k)
        p_ar[k] = 0;

      unsigned int j;
      for (j = 0; j <= p.m_deg; ++j)
      {
        const unsigned int x = p.mp_a[j];
        if (x != 0)
        {
          unsigned int i;
          for (i = 0; i <= deg; ++i)
          {
            const unsigned int y = p_a[i];
            unsigned int t = p_ar[j + i];
            umuladdrem(t, x, y, t, n);
            p_ar[j + i] = t;
          }
        }
      }

      m_deg = degr;
      delete [] mp_a;
      mp_a = p_ar;
    }

    void _Mod()
    {
      unsigned int deg = m_deg;
      unsigned int * const p_a = mp_a;
      while (deg >= m_r)
      {
        p_a[deg - m_r] += p_a[deg];
        if (p_a[deg - m_r] >= m_n) p_a[deg - m_r] -= m_n;
        --deg;

        while (p_a[deg] == 0) --deg;
      }
      m_deg = deg;
    }

    void _Norm()
    {
      const unsigned int deg = m_deg;
      const unsigned int n = m_n;
      unsigned int * const p_a = mp_a;
      if (p_a[deg] != 1)
      {
        const unsigned int y = Inv(p_a[deg], m_n);
        unsigned int i;
        for (i = 0; i <= deg; ++i)
        {
          unsigned int t = p_a[i];
          umulrem(t, t, y, n);
          p_a[i] = t;
        }
      }
    }

  public:
    CPolyMod & operator = (const CPolyMod & p)
    {
      if (&p == this) return *this;
      m_deg = p.m_deg;
      delete [] mp_a;
      mp_a = new unsigned int [p.m_deg + 1];
      unsigned int i;
      for (i = 0; i <= p.m_deg; ++i)
        mp_a[i] = p.mp_a[i];
      return *this;
    }

    int operator != (const CPolyMod & p) const
    {
      if (m_deg != p.m_deg)
        return true;
      unsigned int i;
      for (i = 0; i <= m_deg; ++i)
        if (mp_a[i] != p.mp_a[i])
          return true;
      return false;
    }

    CPolyMod & operator += (unsigned int i)
    {
      const unsigned int t = i % m_n;
      mp_a[0] += t;
      if (mp_a[0] >= m_n) mp_a[0] -= m_n;
      return *this;
    }

    CPolyMod & operator -= (unsigned int i)
    {
      const unsigned int t = m_n - i % m_n;
      mp_a[0] += t;
      if (mp_a[0] >= m_n) mp_a[0] -= m_n;
      return *this;
    }

    CPolyMod Pow(unsigned int e) const
    {
      unsigned int er = 1;
      unsigned int j;
      for (j = e; j != 1; j /= 2)
      {
        er = 2 * er + (j % 2);
      }

      CPolyMod t(*this);
      unsigned int i;
      for (i = er; i != 1; i /= 2)
      {
        t._polySquare();
        t._Mod();
        if (i % 2 != 0)
        {
          t._polyMul(*this);
          t._Mod();
        }
      }
      t._Norm();
      return t;
    }
  };


}
