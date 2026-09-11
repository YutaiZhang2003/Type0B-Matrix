#pragma once
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <gmp.h>
#include <iomanip>
#include <mpc.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
namespace ramond {
inline void require(bool okay, const std::string &message) {
    if (!okay)
        throw std::runtime_error(message);
}
class MP {
    mpc_t value_;
    bool valid_ = true;
    void init() {
        mpc_init2(value_, bits);
        valid_ = true;
    }

  public:
    static inline unsigned digits = 40;
    static inline mpfr_prec_t bits = 136;
    static void precision(unsigned d) {
        require(d >= 30, "higher precision requires at least 30 digits");
        digits = d;
        bits = static_cast<mpfr_prec_t>(std::llround((d + 1.0) * std::log2(10.)));
    }
    MP() {
        init();
        mpc_set_ui(value_, 0, MPC_RNDNN);
    }
    MP(int x) : MP(static_cast<long>(x)) {}
    MP(long x) {
        init();
        mpc_set_si(value_, x, MPC_RNDNN);
    }
    MP(double x) {
        init();
        mpc_set_d(value_, x, MPC_RNDNN);
    }
    MP(std::complex<double> x) {
        init();
        mpc_set_d_d(value_, x.real(), x.imag(), MPC_RNDNN);
    }
    explicit MP(const std::string &s) {
        init();
        if (mpfr_set_str(mpc_realref(value_), s.c_str(), 10, MPFR_RNDN)) {
            mpc_clear(value_);
            valid_ = false;
            throw std::invalid_argument("invalid number: " + s);
        }
        mpfr_set_zero(mpc_imagref(value_), 0);
    }
    MP(const MP &a) {
        init();
        mpc_set(value_, a.value_, MPC_RNDNN);
    }
    MP(MP &&a) noexcept : valid_(a.valid_) {
        std::memcpy(value_, a.value_, sizeof(value_));
        a.valid_ = false;
    }
    ~MP() {
        if (valid_)
            mpc_clear(value_);
    }
    MP &operator=(const MP &a) {
        if (this != &a) {
            if (!valid_)
                init();
            mpc_set(value_, a.value_, MPC_RNDNN);
        }
        return *this;
    }
    MP &operator=(MP &&a) noexcept {
        if (this != &a) {
            if (valid_)
                mpc_clear(value_);
            std::memcpy(value_, a.value_, sizeof(value_));
            valid_ = a.valid_;
            a.valid_ = false;
        }
        return *this;
    }
    mpc_ptr data() {
        return value_;
    }
    mpc_srcptr data() const {
        return value_;
    }
    MP &operator+=(const MP &b) {
        mpc_add(value_, value_, b.value_, MPC_RNDNN);
        return *this;
    }
    MP &operator-=(const MP &b) {
        mpc_sub(value_, value_, b.value_, MPC_RNDNN);
        return *this;
    }
    MP &operator*=(const MP &b) {
        mpc_mul(value_, value_, b.value_, MPC_RNDNN);
        return *this;
    }
    MP &operator/=(const MP &b) {
        require(!b.zero(), "division by zero; generic parameters or a specified limit required");
        mpc_div(value_, value_, b.value_, MPC_RNDNN);
        return *this;
    }
    bool zero() const {
        return mpfr_zero_p(mpc_realref(value_)) && mpfr_zero_p(mpc_imagref(value_));
    }
    friend MP operator+(MP a, const MP &b) {
        return a += b;
    }
    friend MP operator-(MP a, const MP &b) {
        return a -= b;
    }
    friend MP operator*(MP a, const MP &b) {
        return a *= b;
    }
    friend MP operator/(MP a, const MP &b) {
        return a /= b;
    }
    MP operator-() const {
        MP r;
        mpc_neg(r.value_, value_, MPC_RNDNN);
        return r;
    }
    friend bool operator==(const MP &a, const MP &b) {
        return mpc_cmp(a.value_, b.value_) == 0;
    }
    friend bool operator!=(const MP &a, const MP &b) {
        return !(a == b);
    }
};
using Machine = std::complex<double>;
inline MP conj(const MP &a) {
    MP r;
    mpc_conj(r.data(), a.data(), MPC_RNDNN);
    return r;
}
inline Machine conjugate(Machine a) {
    return std::conj(a);
}
inline MP conjugate(const MP &a) {
    return conj(a);
}
inline Machine root(Machine a) {
    return std::sqrt(a);
}
inline MP root(const MP &a) {
    MP r;
    mpc_sqrt(r.data(), a.data(), MPC_RNDNN);
    return r;
}
inline Machine real_number(Machine a) {
    return a.real();
}
inline MP real_number(const MP &a) {
    MP r;
    mpfr_set(mpc_realref(r.data()), mpc_realref(a.data()), MPFR_RNDN);
    return r;
}
inline Machine abs_number(Machine a) {
    return std::abs(a);
}
inline MP abs_number(const MP &a) {
    MP r;
    mpc_abs(mpc_realref(r.data()), a.data(), MPFR_RNDN);
    return r;
}
inline Machine machine(Machine a) {
    return a;
}
inline Machine machine(const MP &a) {
    return {mpfr_get_d(mpc_realref(a.data()), MPFR_RNDN),
            mpfr_get_d(mpc_imagref(a.data()), MPFR_RNDN)};
}
inline double magnitude(Machine a) {
    return std::abs(a);
}
inline double magnitude(const MP &a) {
    return machine(abs_number(a)).real();
}
inline bool finite(Machine a) {
    return std::isfinite(a.real()) && std::isfinite(a.imag());
}
inline bool finite(const MP &a) {
    return mpfr_number_p(mpc_realref(a.data())) && mpfr_number_p(mpc_imagref(a.data()));
}
template <class S> S power(S a, int n) {
    if (n < 0)
        return S(1) / power(a, -n);
    S r(1);
    while (n) {
        if (n & 1)
            r *= a;
        n >>= 1;
        if (n)
            a *= a;
    }
    return r;
}
template <class S> S rational(long a, long b) {
    return S(a) / S(b);
}
template <class S> S parse_real(const std::string &s) {
    require(!s.empty() && s.find_first_not_of("0123456789+-./eE") == std::string::npos,
            "invalid real number: " + s);
    auto p = s.find('/');
    S value;
    if (p != std::string::npos) {
        require(s.find('/', p + 1) == std::string::npos, "invalid rational number: " + s);
        S denominator = parse_real<S>(s.substr(p + 1));
        require(denominator != S(0), "zero denominator in input: " + s);
        value = parse_real<S>(s.substr(0, p)) / denominator;
    } else if constexpr (std::is_same_v<S, MP>)
        value = MP(s);
    else {
        size_t end = 0;
        double x = std::stod(s, &end);
        require(end == s.size(), "invalid number: " + s);
        value = S(x);
    }
    require(finite(value), "nonfinite input: " + s);
    return value;
}
template <class S> S parse(const std::string &s) {
    auto p = s.find(',');
    if (p == std::string::npos)
        return parse_real<S>(s);
    require(s.find(',', p + 1) == std::string::npos, "complex input must be real,imaginary");
    return parse_real<S>(s.substr(0, p)) + S(Machine(0, 1)) * parse_real<S>(s.substr(p + 1));
}
template <class S> bool small(const S &a, double tolerance, int exponent) {
    if constexpr (std::is_same_v<S, MP>) {
        MP x = abs_number(a), t = power(MP(10), exponent);
        return mpfr_cmp(mpc_realref(x.data()), mpc_realref(t.data())) <= 0;
    } else
        return magnitude(a) <= tolerance;
}
template <class S> int digits() {
    if constexpr (std::is_same_v<S, MP>)
        return MP::digits;
    else
        return 0;
}
template <class S> bool arithmetic_zero(const S &a) {
    return small(a, 1e-13, -std::max(20, digits<S>() - 20));
}
inline std::string decimal(mpfr_srcptr x) {
    char *s = nullptr;
    mpfr_asprintf(&s, "%.*Rg", MP::digits + 2, x);
    std::string r(s);
    mpfr_free_str(s);
    return r;
}
inline std::string json_number(Machine a) {
    require(finite(a), "nonfinite coefficient");
    std::ostringstream out;
    out << std::setprecision(17) << "{\"real\":\"" << a.real() << "\",\"imag\":\"" << a.imag()
        << "\"}";
    return out.str();
}
inline std::string json_number(const MP &a) {
    require(finite(a), "nonfinite coefficient");
    return "{\"real\":\"" + decimal(mpc_realref(a.data())) + "\",\"imag\":\"" +
           decimal(mpc_imagref(a.data())) + "\"}";
}
} // namespace ramond
