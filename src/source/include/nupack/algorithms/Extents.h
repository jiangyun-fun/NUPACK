/**
 * @brief begin_of, end_of, and len operations - offers a bit more generality than std:: versions
 *
 * @file Extents.h
 * @author Mark Fornace
 * @date 2018-05-18
 */
#pragma once
#include "TypeSupport.h"
#include "Macro.h"
#include <iterator>

/******************************************************************************************/

namespace nupack_adl {
    using std::begin;
    using std::end;
    using std::size;
    template <class T>
    auto begin_of(T &&t) -> decltype(begin(std::forward<T>(t))) {return begin(std::forward<T>(t));}
    template <class T>
    auto end_of(T &&t) -> decltype(end(std::forward<T>(t))) {return end(std::forward<T>(t));}
    template <class T>
    auto size_of(T &&t) -> decltype(size(std::forward<T>(t))) {return size(std::forward<T>(t));}
}

/******************************************************************************************/

namespace nupack {

NUPACK_UNARY_FUNCTOR(begin_of, nupack_adl::begin_of(fw<T>(t)));
NUPACK_UNARY_FUNCTOR(end_of, nupack_adl::end_of(fw<T>(t)));
NUPACK_UNARY_FUNCTOR(len, nupack_adl::size(fw<T>(t)));

NUPACK_DETECT(has_len, decltype(len(declval<T>())));

NUPACK_DETECT(is_random_access, decltype(begin_of(declref<T>()) + 2));

NUPACK_DETECT(is_iterable, decltype(
    begin_of(declref<T>()) != end_of(declref<T>()),   // begin/end and operator !=
    ++declref<decltype(begin_of(declref<T>()))>(), // operator ++
    *begin_of(declref<T>())                        // dereference operator
));

/******************************************************************************************/

}
