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

template < typename dtype, unsigned int dimState >
using solveStep1D_ftype = std::function< void(const Field1D<dtype, dimState>&, Field1D<dtype, dimState>&, float) >;

template < typename dtype, unsigned int dimState >
using solveStep2D_ftype = std::function< void(Field2D<dtype, dimState>&, Field2D<dtype, dimState>&, float) >;

/*#####################################
#--------- Boundary handler --------- #
#####################################*/

namespace BoundaryHandler {
    
    template < typename dtype, unsigned int dimState >
    void apply(Field1D<dtype, dimState>& Q_field, int bc_left, int bc_right) {
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
template < typename dtype, unsigned int dimState >
class FiniteVolumeSolver {
    public:
        // ATTRIBUTES
        const FluxMaker<dtype, dimState>& fmaker;

        FiniteVolumeSolver(const FluxMaker<dtype, dimState>& fmaker): fmaker(fmaker) {}

        // METHODS
        /** @brief Returns initialized fields Q and Q_next - Field1D*/
        std::tuple< Field1D<dtype, dimState>, Field1D<dtype, dimState> > initialize(
                const Mesh1D& mesh, const fnX_ftype<dtype, dimState>& f_init, 
                int bc_left = 0, int bc_right = 0) const {

            Field1D<dtype, dimState> Q = mesh.evaluate_center<dtype, dimState>(f_init);
            Field1D<dtype, dimState> Q_next = mesh.evaluate_center<dtype, dimState>(f_init);

            BoundaryHandler::apply<dtype, dimState>(Q, bc_left, bc_right);
            BoundaryHandler::apply<dtype, dimState>(Q_next, bc_left, bc_right);

            return {Q, Q_next};
        }

        /** @brief Returns the solve step - Mesh1D*/
        solveStep1D_ftype<dtype, dimState> get_solve_step(
                const ModelMaker<dtype, dimState>& model_maker, const Mesh1D& mesh, int bc_left = 0, int bc_right = 0) {

            float dx = mesh.get_dx();
            size_t nCx = mesh.get_nCx();
            
            Model<dtype, dimState> model_x = model_maker.make_normal_model();
            NumericalFlux<dtype, dimState> F = fmaker.make_numflux( model_x );

            solveStep1D_ftype<dtype, dimState> solve_step = [F, dx, nCx, bc_right, bc_left]
                (const Field1D<dtype, dimState>& Q, Field1D<dtype, dimState>& Q_next, float dt) {
                
                // Main update loop
                #pragma omp parallel for shared( Q_next )
                for (size_t i = 1; i <= nCx; i ++) {
                    Q_next(i) = Q(i) - ( F.num_flux(Q(i), Q(i+1)) 
                                       - F.num_flux(Q(i-1), Q(i)) )*(dt/dx);
                }
                
                // Apply bc conditions
                BoundaryHandler::apply<dtype, dimState>(Q_next, bc_left, bc_right);
            };

            return solve_step;
        }
};

#endif