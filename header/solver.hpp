#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <iostream>
#include <functional>
#include <memory>
#include <string>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <omp.h>

#include "mesh/mesh1D.hpp"
#include "mesh/mesh2D.hpp"
#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
#include "models.hpp"
using namespace FVTYPES;

template < typename dtype, size_t dimState >
using solveStep1D_ftype = std::function< void(const Field1D<dtype, dimState>&, Field1D<dtype, dimState>&, float) >;

template < typename dtype, size_t dimState >
using solveStep2D_ftype = std::function< void(const Field2D<dtype, dimState>&, Field2D<dtype, dimState>&, float) >;

template < typename fieldType >
using solveStep_ftype = std::function< void(const fieldType&, fieldType&, float) >;

/*#####################################
#--------- Numerical fluxes --------- #
#####################################*/

namespace NumFlux 
{
    template < typename dtype, size_t dimState >
    inline Var<dtype, dimState> GodunovFlux( const Var<dtype, dimState>& QL, const Var<dtype, dimState>& QR, const Model<dtype, dimState>& model ) {
        return model.flux( model.riemann_solver(QL, QR, 0) );
    }

    template < typename dtype, size_t dimState >
    inline Var<dtype, dimState> RusanovFlux( const Var<dtype, dimState>& QL, const Var<dtype, dimState>& QR, const Model<dtype, dimState>& model ) {
        return ( model.flux(QL)+model.flux(QR) )*0.5
             - (QR - QL)*std::max( model.wave_speed(QL), model.wave_speed(QL) )*0.5;
    }
}

/*#####################################
#--------- Boundary handler --------- #
#####################################*/

namespace BoundaryHandler {
    
    template < typename dtype, size_t dimState >
    void apply(Field1D<dtype, dimState>& Q_field, int bc_left, int bc_right) {
        size_t nx = Q_field.get_nx();
        
        if (bc_left == 0) Q_field(0) = Q_field(1);
        else Q_field(0) = Q_field(nx);

        if (bc_right== 1) Q_field(nx+1) = Q_field(nx);
        else Q_field(nx+1) = Q_field(1);
    }

    template < typename dtype, size_t dimState >
    void apply(Field2D<dtype, dimState>& Q_field, int bc_left, int bc_right, int bc_up, int bc_down) {
        size_t nx = Q_field.get_nx();
        size_t ny = Q_field.get_ny();

        for (size_t j = 1; j <= ny; ++ j) {
            if (bc_left == 0) Q_field(0,j) = Q_field(1,j);
            else Q_field(0,j) = Q_field(nx,j);
        
            if (bc_right == 0) Q_field(nx+1,j) = Q_field(nx,j);
            else Q_field(nx+1,j) = Q_field(1,j);
        }

        for (size_t i = 1; i <= nx; ++ i) {
            if (bc_down == 0) Q_field(i,0) = Q_field(i,1);
            else Q_field(i,0) = Q_field(i,ny);

            if (bc_up == 0) Q_field(i,ny+1) = Q_field(i,ny);
            else Q_field(i,ny+1) = Q_field(i,1); 
        }
    }
}

/*#########################################
#--------- Finite volume solver --------- #
#########################################*/

/** @brief Finite volume solver*/
template < typename dtype, size_t dimState >
class FiniteVolumeSolver {
    public:
        // ATTRIBUTES
        using edgeFlux = Var<dtype, dimState> (*)( const Var<dtype, dimState>&, const Var<dtype, dimState>&, const Model<dtype, dimState>& );
        
        edgeFlux F_num;

        const ModelMaker<dtype, dimState>& model_maker;
        std::vector< std::unique_ptr< Model<dtype, dimState> > > models;

        // CONSTRUCTOR
        FiniteVolumeSolver(const ModelMaker<dtype, dimState>& model_maker,
                           const std::string& flux_type): model_maker(model_maker) {
            if (flux_type == "godunov") {
                F_num = NumFlux::GodunovFlux<dtype, dimState>;
            } else {
                F_num = NumFlux::RusanovFlux<dtype, dimState>;
            }
        }

        // METHODS
        /** @brief Returns initialized fields Q and Q_next - Field1D*/
        Field1D<dtype, dimState> initialize(
                const Mesh1D& mesh, const fnX_ftype<dtype, dimState>& f_init, 
                int bc_left = 0, int bc_right = 0) const {

            Field1D<dtype, dimState> Q = mesh.evaluate_center<dtype, dimState>(f_init);
            BoundaryHandler::apply<dtype, dimState>(Q, bc_left, bc_right);

            return Q;
        }

        /** @brief Returns the solve step - Mesh1D*/
        solveStep1D_ftype<dtype, dimState> get_solve_step(const Mesh1D& mesh, int bc_left = 0, int bc_right = 0) {
            // Get mesh dimensions
            float dx = mesh.get_dx();
            size_t nCx = mesh.get_nCx();
            
            // Build and store 1D models
            models.push_back( std::move( model_maker.make_normal_model() ) );
            auto* mx = models[0].get();

            solveStep1D_ftype<dtype, dimState> solve_step = [=]
                (const Field1D<dtype, dimState>& Q, Field1D<dtype, dimState>& Q_next, float dt) {
                
                const auto cdx = dt/dx;

                // Main update loop
                #pragma omp parallel for schedule(static)
                for (size_t i = 1; i <= nCx; ++ i) {
                    #pragma omp simd
                    for (int k = 0; k < 1; ++ k) {
                        const auto qi = Q(i);
                        Q_next(i) = qi - (F_num( Q(i), Q(i+1), *mx ) - 
                                          F_num( Q(i-1), Q(i), *mx ))*cdx;
                    }
                }
                
                // Apply bc conditions
                BoundaryHandler::apply<dtype, dimState>(Q_next, bc_left, bc_right);
            };

            return solve_step;
        }

        /** @brief Returns initialized fields Q and Q_next - Field2D*/
        Field2D<dtype, dimState> initialize(
                const Mesh2D& mesh, const fnXY_ftype<dtype, dimState>& f_init, 
                int bc_left = 0, int bc_right = 0, int bc_up = 0, int bc_down = 0) const {

            Field2D<dtype, dimState> Q = mesh.evaluate_center<dtype, dimState>(f_init);
            BoundaryHandler::apply<dtype, dimState>(Q, bc_left, bc_right, bc_up, bc_down);

            return Q;
        }

        /** @brief Returns the solve step - Mesh2D*/
        solveStep2D_ftype<dtype, dimState> get_solve_step(const Mesh2D& mesh, int bc_left = 0, int bc_right = 0, int bc_up = 0, int bc_down = 0) {
            // Get mesh dimensions
            float dx = mesh.get_dx(); float dy = mesh.get_dy();
            size_t nCx = mesh.get_nCx(); size_t nCy = mesh.get_nCy();
            
            // Build and store 2D models
            models.push_back( std::move( model_maker.make_normal_model({1.0, 0.0}) ) );
            models.push_back( std::move( model_maker.make_normal_model({0.0, 1.0}) ) );
            auto* mx = models[0].get();
            auto* my = models[0].get();
            
            solveStep2D_ftype<dtype, dimState> solve_step = [=]
                (const Field2D<dtype, dimState>& Q, Field2D<dtype, dimState>& Q_next, float dt) {
                
                const auto cdx = dt/dx;
                const auto cdy = dt/dy;
                
                // Main update loop
                #pragma omp parallel for schedule(static)
                for (size_t j = 1; j <= nCy; j ++) {
                    #pragma omp simd
                    for (size_t i = 1; i <= nCx; i ++) {
                        const auto qij = Q(i,j);

                        Q_next(i,j) = qij - (F_num( Q(i,j), Q(i+1,j), *mx ) - 
                                             F_num( Q(i-1,j), Q(i,j), *mx ))*cdx
                                          - (F_num( Q(i,j), Q(i,j+1), *my ) -
                                             F_num( Q(i,j-1), Q(i,j), *my ))*cdy;
                    }
                }
                
                // Apply bc conditions
                BoundaryHandler::apply<dtype, dimState>(Q_next, bc_left, bc_right, bc_up, bc_down);
            };

            return solve_step;
        }
};

namespace Integrator 
{

template < typename fieldType >
fieldType Euler( const solveStep_ftype<fieldType>& solve_step, 
                 const fieldType& Q_init, 
                 float t_final, float dt ) {
    float time = 0.0;

    fieldType Q( Q_init );
    fieldType Q_next( Q_init );

    while (time < t_final) {

        /*if (time + dt < t_final) {
            time += dt;
        } else {
            dt = t_final - time;
            time = t_final;
        }*/

        solve_step(Q, Q_next, dt);
        std::swap(Q, Q_next);
        time += dt;
    }

    return Q;
}

template < typename fieldType >
fieldType RK2( const solveStep_ftype<fieldType>& solve_step, 
               const fieldType& Q_init, 
               float t_final, float dt ) {
    float time = 0.0;

    fieldType Q( Q_init );
    fieldType Q_next( Q_init );
    fieldType Q_inter( Q_init );

    while (time < t_final) {
        solve_step(Q, Q_inter, dt);

        if (time + dt < t_final) {
            time += dt;
        } else {
            dt = t_final - time;
            time = t_final;
        }

        solve_step(Q_inter, Q_next);
        Q = Q*0.5 + Q_next*0.5;
    }

    return Q;
}
}

#endif