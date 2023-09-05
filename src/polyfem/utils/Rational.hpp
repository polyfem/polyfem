#pragma once

#include <gmp.h>

#include <polyfem/Common.hpp>

namespace polyfem
{
	namespace utils
	{
        class Rational {
        public:
            mpq_t value;

            void canonicalize() { mpq_canonicalize(value); }

            int get_sign() const { return mpq_sgn(value); }

            void print_numerator()
            {
                mpz_t numerator;
                mpz_init(numerator);
                mpq_get_num(numerator, value);
                mpz_out_str(NULL, 10, numerator);
                // long v=mpz_get_si(numerator);
                mpz_clear(numerator);
            }

            void print_denominator()
            {
                mpz_t denominator;
                mpz_init(denominator);
                mpq_get_den(denominator, value);

                mpz_out_str(NULL, 10, denominator);
                // long v=mpz_get_si(denominator);
                mpz_clear(denominator);
            }

            long long get_numerator()
            {
                mpz_t numerator;
                mpz_init(numerator);

                mpq_get_num(numerator, value);
                long long v = mpz_get_si(numerator);

                mpz_clear(numerator);
                return v;
            }

            long long get_denominator()
            {
                mpz_t denominator;
                mpz_init(denominator);
                mpq_get_den(denominator, value);

                long long v = mpz_get_si(denominator);
                // std::string s(mpz_get_str(NULL, 10, denominator));
                // long long v=std::stoll(s);
                mpz_clear(denominator);
                return v;
            }

            std::string get_denominator_str()
            {
                mpz_t denominator;
                mpz_init(denominator);
                mpq_get_den(denominator, value);

                std::string v(mpz_get_str(NULL, 10, denominator));

                mpz_clear(denominator);
                return v;
            }

            std::string get_numerator_str()
            {
                mpz_t numerator;
                mpz_init(numerator);

                mpq_get_num(numerator, value);
                std::string v(mpz_get_str(NULL, 10, numerator));
                ;

                mpz_clear(numerator);
                return v;
            }

            double get_double(const std::string &num, const std::string &denom)
            {
                std::string tmp = num + "/" + denom;
                mpq_set_str(value, tmp.c_str(), 10);
                return mpq_get_d(value);
            }

            Rational()
            {
                mpq_init(value);
                mpq_set_d(value, 0);
            }

            Rational(double d)
            {
                assert(std::isfinite(d));
                mpq_init(value);
                mpq_set_d(value, d);
                canonicalize();
            }

            Rational(float d)
            {
                assert(std::isfinite(d));
                mpq_init(value);
                double ddouble = d; // convert (float)d to double
                mpq_set_d(value, ddouble);
                canonicalize();
            }

            Rational(int i)
            {
                mpq_init(value);
                mpq_set_si(value, i, 1U);
                canonicalize();
            }

            Rational(long i)
            {
                mpq_init(value);
                mpq_set_si(value, i, 1U);
                canonicalize();
            }

            Rational(const mpq_t &v_)
            {
                mpq_init(value);
                mpq_set(value, v_);
                // canonicalize();
            }

            Rational(const Rational &other)
            {
                mpq_init(value);
                mpq_set(value, other.value);
            }

            ~Rational() { mpq_clear(value); }

            friend Rational operator-(const Rational &v)
            {
                Rational r_out;
                mpq_neg(r_out.value, v.value);
                return r_out;
            }

            friend Rational operator+(const Rational &x, const Rational &y)
            {
                Rational r_out;
                mpq_add(r_out.value, x.value, y.value);
                return r_out;
            }

            friend Rational operator-(const Rational &x, const Rational &y)
            {
                Rational r_out;
                mpq_sub(r_out.value, x.value, y.value);
                return r_out;
            }

            friend Rational operator*(const Rational &x, const Rational &y)
            {
                Rational r_out;
                mpq_mul(r_out.value, x.value, y.value);
                return r_out;
            }

            friend Rational operator/(const Rational &x, const Rational &y)
            {
                Rational r_out;
                mpq_div(r_out.value, x.value, y.value);
                return r_out;
            }

            Rational &operator=(const Rational &x)
            {
                if (this == &x)
                    return *this;
                mpq_set(value, x.value);
                return *this;
            }

            Rational &operator=(const double x)
            {
                mpq_set_d(value, x);
                // canonicalize();
                return *this;
            }

            Rational &operator=(const float x)
            {
                double xd = x;
                mpq_set_d(value, xd);
                // canonicalize();
                return *this;
            }

            Rational &operator=(const int x)
            {
                mpq_set_si(value, x, 1U);
                // canonicalize();
                return *this;
            }

            Rational &operator=(const long x)
            {
                mpq_set_si(value, x, 1U);
                // canonicalize();
                return *this;
            }

            template <typename T> bool operator<(const T &r1)
            {
                if constexpr (std::is_same<T, Rational>::value)
                    return mpq_cmp(value, r1.value) < 0;
                else
                    return *this < Rational(r1);
            }

            template <typename T> bool operator>(const T &r1)
            {
                if constexpr (std::is_same<T, Rational>::value)
                    return mpq_cmp(value, r1.value) > 0;
                else
                    return *this > Rational(r1);
            }

            template <typename T> bool operator<=(const T &r1)
            {
                if constexpr (std::is_same<T, Rational>::value)
                    return mpq_cmp(value, r1.value) <= 0;
                else
                    return *this <= Rational(r1);
            }

            template <typename T> bool operator>=(const T &r1)
            {
                if constexpr (std::is_same<T, Rational>::value)
                    return mpq_cmp(value, r1.value) >= 0;
                else
                    return *this >= Rational(r1);
            }

            template <typename T> bool operator==(const T &r1)
            {
                if constexpr (std::is_same<T, Rational>::value)
                    return mpq_equal(value, r1.value);
                else
                    return *this == Rational(r1);
            }

            template <typename T> bool operator!=(const T &r1)
            {
                if constexpr (std::is_same<T, Rational>::value)
                    return !mpq_equal(value, r1.value);
                else
                    return *this != Rational(r1);
            }

            double to_double() const { return mpq_get_d(value); }

            operator double() const { return to_double(); }

            friend std::ostream &operator<<(std::ostream &os, const Rational &r)
            {
                os << mpq_get_d(r.value);
                return os;
            }
        };
    }
}