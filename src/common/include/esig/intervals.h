//
// Created by user on 30/03/2022.
//

#ifndef ESIG_INTERVALS_H_
#define ESIG_INTERVALS_H_

#include <esig/esig_export.h>
#include <esig/implementation_types.h>

#include <iosfwd>
#include <limits>
#include <vector>

namespace esig {

enum class ESIG_EXPORT interval_type
{
    clopen,
    opencl
};



class ESIG_EXPORT interval
{
public:

    virtual ~interval() = default;

protected:

    interval_type m_interval_type;

public:

    interval();
    explicit interval(interval_type);

    interval_type get_type() const noexcept;

    virtual bool operator==(const interval& rhs) const;
    virtual bool operator!=(const interval& rhs) const;

    virtual param_t included_end() const;
    virtual param_t excluded_end() const;

    virtual param_t inf() const = 0;
    virtual param_t sup() const = 0;

    virtual bool contains(param_t arg) const noexcept;
    virtual bool is_associated(const interval& arg) const noexcept;
    virtual bool contains(const interval& arg) const noexcept;
    virtual bool intersects_with(const interval& arg) const noexcept;

};

ESIG_EXPORT std::ostream& operator<<(std::ostream&, const interval& itvl);

class ESIG_EXPORT dyadic;


ESIG_EXPORT bool dyadic_equals(const dyadic &lhs, const dyadic &rhs);
ESIG_EXPORT bool rational_equals(const dyadic &lhs, const dyadic &rhs);
ESIG_EXPORT std::ostream &operator<<(std::ostream &os, const dyadic &di);

class ESIG_EXPORT dyadic
{
public:
    using multiplier_t = int;
    using power_t = int;

protected:
    multiplier_t m_multiplier;
    power_t m_power;

public:
    static multiplier_t mod(multiplier_t a, multiplier_t b);
    static multiplier_t int_two_to_int_power(power_t exponent);
    static multiplier_t shift(multiplier_t k, power_t n);


    multiplier_t multiplier() const noexcept;
    power_t power() const noexcept;

    dyadic() = default;

    explicit dyadic(multiplier_t k);
    dyadic(multiplier_t k, power_t n);

    explicit operator param_t() const;

public:

    dyadic& move_forward(multiplier_t arg);
    dyadic& operator++();
    const dyadic operator++(int);
    dyadic& operator--();
    const dyadic operator--(int);

public:
    bool rebase(power_t resolution=std::numeric_limits<power_t>::lowest());

public:
    bool operator<(const dyadic& rhs) const;
    bool operator>(const dyadic& rhs) const;
    bool operator<=(const dyadic& rhs) const;
    bool operator>=(const dyadic& rhs) const;

    friend bool dyadic_equals(const dyadic& lhs, const dyadic& rhs);
    friend bool rational_equals(const dyadic& lhs, const dyadic& rhs);

public:

    friend std::ostream& operator<<(std::ostream& os, const dyadic& di);

};




class ESIG_EXPORT dyadic_interval;


ESIG_EXPORT std::vector<dyadic_interval>
to_dyadic_intervals(param_t inf, param_t sup, dyadic::power_t tol, interval_type itype = interval_type::clopen);


class ESIG_EXPORT dyadic_interval : public dyadic, public interval
{

public:

    using typename dyadic::multiplier_t;
    using typename dyadic::power_t;
private:
    using dyadic::m_multiplier;
    using dyadic::m_power;

public:
    using dyadic::operator++;
    using dyadic::operator--;
    using dyadic::multiplier;
    using dyadic::power;

public:

    dyadic_interval();
    explicit dyadic_interval(multiplier_t k);
    dyadic_interval(multiplier_t k, power_t n);
    dyadic_interval(multiplier_t k, power_t n, interval_type itype);
    explicit dyadic_interval(interval_type itype);
    explicit dyadic_interval(dyadic di);
    dyadic_interval(dyadic di, power_t resolution, interval_type itype=interval_type::clopen);
    dyadic_interval(param_t val, power_t resolution, interval_type itype=interval_type::clopen);
    dyadic_interval(dyadic di, interval_type itype);


    multiplier_t unit() const noexcept;


    param_t included_end() const override;
    param_t excluded_end() const override;
    param_t inf() const override;
    param_t sup() const override;

    dyadic dincluded_end() const;
    dyadic dexcluded_end() const;
    dyadic dinf() const;
    dyadic dsup() const;

    dyadic_interval shrink_to_contained_end(power_t arg=1) const;
    dyadic_interval shrink_to_omitted_end() const;
    dyadic_interval& shrink_interval_right();
//    dyadic_interval& shrink_interval_left();
    dyadic_interval& shrink_interval_left(power_t k=1);
    dyadic_interval& expand_interval(multiplier_t arg=1);
    bool contains(const dyadic_interval& other) const;
    bool aligned() const;
    dyadic_interval& flip_interval();
    dyadic_interval shift_forward(power_t arg=1) const;
    dyadic_interval shift_back(power_t arg=1) const;

    dyadic_interval& advance() noexcept;

    bool operator==(const interval& other) const override;


    friend std::vector<dyadic_interval> to_dyadic_intervals(param_t inf, param_t sup, power_t resolution, interval_type itype);
};

ESIG_EXPORT std::ostream& operator<<(std::ostream& os, const dyadic_interval& di);

class ESIG_EXPORT real_interval : public interval
{
    param_t m_inf;
    param_t m_sup;

public:

    real_interval();

    real_interval(const real_interval& other) = default;
    real_interval(real_interval&& other) noexcept = default;

    explicit real_interval(interval_type itype);
    real_interval(param_t inf, param_t sup);
    real_interval(param_t inf, param_t sup, interval_type itype);
    explicit real_interval(const interval& itvl);

    real_interval& operator=(const real_interval& other) = default;
    real_interval& operator=(real_interval&& other) noexcept = default;

    param_t included_end() const override;
    param_t excluded_end() const override;
    param_t inf() const override;
    param_t sup() const override;
    bool contains(param_t arg) const noexcept override;
};




class ESIG_EXPORT partition : public real_interval {

    std::vector<param_t> m_midpoints;
public:

    using real_interval::real_interval;

    partition(real_interval base, std::initializer_list<param_t> midpoints);

    real_interval operator[](dimn_t index) noexcept;

};


} // namespace esig


#endif//ESIG_INTERVALS_H_
