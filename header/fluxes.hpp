#ifndef FLUXES_HPP
#define FLUXES_HPP

#include <iostream>
#include <functional>
#include <algorithm>
#include <cmath>

#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
#include "models.hpp"
using namespace FVTYPES;

/*###################################
#--------- Numerical Flux --------- #
###################################*/

/** @brief Numerical flux container*/
template < typename dtype, unsigned int dimState >
struct NumericalFlux {
    private:
        static const unsigned int dim = dimState;
    
    public:
        numEdgeFlux_ftype<dtype, dimState> num_flux;
        NumericalFlux(numEdgeFlux_ftype<dtype, dimState> num_flux): num_flux(num_flux) {}
};

/*###############################
#--------- Flux Maker --------- #
###############################*/

/** @brief Abstrcat flux maker*/
template < typename dtype, unsigned int dimState >
struct FluxMaker {
    virtual NumericalFlux<dtype, dimState> make_numflux(const Model<dtype, dimState>&) const = 0;
};

/** @brief Rusanov (LLF) numerical flux*/
template < typename dtype, unsigned int dimState >
struct RusanovFlux: FluxMaker<dtype, dimState> {
    NumericalFlux<dtype, dimState> make_numflux(const Model<dtype, dimState>& model) const {
        numEdgeFlux_ftype<dtype, dimState> num_flux = [model](const StateType::Var<dtype, dimState>& QL,
                                                               const StateType::Var<dtype, dimState>& QR) {
            return ( model.flux(QL)+model.flux(QR) )*0.5
                 - (QR - QL)*std::max( model.wave_speed(QL), model.wave_speed(QL) )*0.5;
        };
        return NumericalFlux<dtype, dimState>(num_flux);
    }
};

/** @brief Godunov numerical flux*/
template < typename dtype, unsigned int dimState >
struct GodunovFlux: FluxMaker<dtype, dimState> {
    NumericalFlux<dtype, dimState> make_numflux(const Model<dtype, dimState>& model) const {
        numEdgeFlux_ftype<dtype, dimState> num_flux = [model](const StateType::Var<dtype, dimState>& QL,
                                                               const StateType::Var<dtype, dimState>& QR) {
            return model.flux( model.riemann_solver(QL, QR, 0.0) );
        };
        return num_flux;
    }
};

#endif