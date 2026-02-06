#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <iostream>
#include <cmath>
#include <initializer_list>
#include <type_traits>
#include <functional>
#include <stdexcept>
#include <complex>
#include <algorithm>

namespace FVTYPES {

    using xPoint   = double;
    using xyPoint  = std::array<double, 2>;
    using xyzPoint = std::array<double, 3>;

    template < typename dtype, size_t dimState >
    using Var = std::conditional_t<dimState == 1, dtype, std::array<dtype, dimState>>;

    template < typename dtype, size_t dimState >
    using numEdgeFlux_ftype = std::function< Var<dtype, dimState>(const Var<dtype, dimState>&, const Var<dtype, dimState>&) >;

    template < typename dtype, size_t dimState >
    using fnX_ftype = std::function< Var<dtype, dimState>(double) >;

    template < typename dtype, size_t dimState >
    using fnXY_ftype = std::function< Var<dtype, dimState>( xyPoint ) >;

    template < typename dtype, size_t dimState >
    using fnXYZ_ftype = std::function< Var<dtype, dimState>( xyzPoint ) >;
}

#endif