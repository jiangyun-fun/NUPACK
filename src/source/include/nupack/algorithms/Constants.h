/**
 * @brief Compile-time constants; pretty rudimentary and should be swapped out (maybe for boost hana)
 *
 * @file Constants.h
 * @author Mark Fornace
 * @date 2018-05-18
 */
#pragma once
#include "Traits.h"
#include <limits>

namespace nupack {

/******************************************************************************************/

template <int N>
using size_constant = std::integral_constant<std::size_t, N>;

template <int N>
using int_constant = std::integral_constant<int, N>;

/******************************************************************************************/

template <class T>
struct Always {
    T t;
    template <class ...Ts>
    constexpr T operator()(Ts const &...ts) const {return t;}
};

using False = std::false_type;
using True = std::true_type;

template <bool B>
using bool_t  = std::conditional_t<B, True, False>;

template <class T, class SFINAE=void>
struct infinity_t {
    constexpr T operator()() const {
        if constexpr(std::is_floating_point_v<T>) return std::numeric_limits<T>::infinity();
        else return std::numeric_limits<T>::max();
    }
};

template <class T, class SFINAE=void>
struct nan_t {
    constexpr T operator()() const {
        static_assert(std::is_floating_point_v<T>);
        return std::numeric_limits<T>::quiet_NaN();
    }
};

template <class T>
constexpr T inf() {return infinity_t<T>()();}

template <class T>
constexpr T nan() {return nan_t<T>()();}

/******************************************************************************************/

}
