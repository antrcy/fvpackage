#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <iostream>
#include <functional>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <omp.h>

#include "mesh/mesh1D.hpp"
#include "mesh/mesh2D.hpp"
#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
#include "fluxes.hpp"
using namespace FVTYPES;

template < unsigned int dimState >
using solveStep1D_ftype = std::function< void(const Field1D<dimState>&, Field1D<dimState>&, float) >;

template < unsigned int dimState >
using solveStep2D_ftype = std::function< void(Field2D<dimState>&, Field2D<dimState>&, float) >;

/*#####################################
#--------- Boundary handler --------- #
#####################################*/

namespace BoundaryHandler { 
    
    template < unsigned int dimState >
    void apply(Field1D<dimState>& Q_field, int bc_left, int bc_right) {
        size_t nx = Q_field.get_nx();
        
        if (bc_left == 0) Q_field(0) = Q_field(1);
        else Q_field(0) = Q_field(nx);

        if (bc_right== 1) Q_field(nx+1) = Q_field(nx);
        else Q_field(nx+1) = Q_field(1);
    }
}

/*#########################################
#--------- Finite volume solver --------- #
#########################################*/

/** @brief Finite volume solver*/
template < unsigned int dimState >
class FiniteVolumeSolver {
    public:
        // ATTRIBUTES
        const FluxMaker<dimState>* flux_maker;

        FiniteVolumeSolver(const FluxMaker<dimState>& flux_maker): flux_maker(&flux_maker) {}

        // METHODS
        /** @brief Returns initialized fields Q and Q_next - Field1D*/
        std::tuple< Field1D<dimState>, Field1D<dimState> > initialize(
                const Mesh1D& mesh, const fn_x_ftype<dimState>& f_init, 
                int bc_left = 0, int bc_right = 0) const {

            Field1D<dimState> Q = mesh.evaluate_center<dimState>(f_init);
            Field1D<dimState> Q_next = mesh.evaluate_center<dimState>(f_init);

            BoundaryHandler::apply<dimState>(Q, bc_left, bc_right);
            BoundaryHandler::apply<dimState>(Q_next, bc_left, bc_right);

            return {Q, Q_next};
        }

        /** @brief Returns the solve step - Mesh1D*/
        solveStep1D_ftype<dimState> get_solve_step(
                const Mesh1D& mesh, const Model<dimState>& model,
                int bc_left = 0, int bc_right = 0) const {

            float dx = mesh.get_dx();
            size_t nCx = mesh.get_nCx();

            NumericalFlux<dimState> F = flux_maker->make_numflux( model );
            
            solveStep1D_ftype<dimState> solve_step = [dx, nCx, F, bc_right, bc_left]
                (const Field1D<dimState>& Q, Field1D<dimState>& Q_next, float dt) {
                
                // Main update loop
                #pragma omp parallel for shared( Q_next )
                for (size_t i = 1; i <= nCx; i ++) {
                    Q_next(i) = Q(i) - ( F.num_flux(Q(i), Q(i+1)) - F.num_flux(Q(i-1), Q(i)) )*(dt/dx);
                }
                
                // Apply bc conditions
                BoundaryHandler::apply<dimState>(Q_next, bc_left, bc_right);
            };

            return solve_step;
        }
};

#endif