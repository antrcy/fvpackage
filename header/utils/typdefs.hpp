#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <iostream>
#include <functional>
#include <algorithm>

#include "linear.hpp"

namespace FVTYPES {
    using index_t = unsigned long int;

    using xPoint   = double;
    using xyPoint  = std::array<double, 2>;
    using xyzPoint = std::array<double, 3>;

    template < typename dtype, unsigned int dimState >
    using flux_ftype = std::function< StateType::Var<dtype, dimState>(const StateType::Var<dtype, dimState>&) >;

    template < typename dtype, unsigned int dimState >
    using waveSpeed_ftype = std::function< dtype(const StateType::Var<dtype, dimState>&) >;

    template < typename dtype, unsigned int dimState >
    using riemannSolver_ftype = std::function< StateType::Var<dtype, dimState>(const StateType::Var<dtype, dimState>&, const StateType::Var<dtype, dimState>&, dtype) >;

    template < typename dtype, unsigned int dimState >
    using numEdgeFlux_ftype = std::function< StateType::Var<dtype, dimState>(const StateType::Var<dtype, dimState>&, const StateType::Var<dtype, dimState>&) >;

    template < typename dtype, unsigned int dimState >
    using fnX_ftype = std::function< StateType::Var<dtype, dimState>(double) >;

    template < typename dtype, unsigned int dimState >
    using fnXY_ftype = std::function< StateType::Var<dtype, dimState>( xyPoint ) >;

    template < typename dtype, unsigned int dimState >
    using fnXYZ_ftype = std::function< StateType::Var<dtype, dimState>( xyzPoint ) >;
}

#endif