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

    template < unsigned int dimState >
    using flux_ftype = std::function< StateType::Var<dimState>(const StateType::Var<dimState>&) >;

    template < unsigned int dimState >
    using wave_speed_ftype = std::function< double(const StateType::Var<dimState>&) >;

    template < unsigned int dimState >
    using riemann_solver_ftype = std::function< StateType::Var<dimState>(const StateType::Var<dimState>&, const StateType::Var<dimState>&, double) >;

    template < unsigned int dimState >
    using numEdgeFlux_ftype = std::function< StateType::Var<dimState>(const StateType::Var<dimState>&, const StateType::Var<dimState>&) >;

    template < unsigned int dimState >
    using fn_x_ftype = std::function< StateType::Var<dimState>(double) >;

    template < unsigned int dimState >
    using fn_xy_ftype = std::function< StateType::Var<dimState>( xyPoint ) >;

    template < unsigned int dimState >
    using fn_xyz_ftype = std::function< StateType::Var<dimState>( xyzPoint ) >;
}

#endif