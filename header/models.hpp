#ifndef MODELS_HPP
#define MODELS_HPP

#include <iostream>
#include <functional>
#include <algorithm>
#include <memory>
#include <cmath>

#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
using namespace FVTYPES;

/*##########################
#--------- Model --------- #
###########################*/

/**@brief One dimensional conservation model*/
template < typename dtype, unsigned int dimState >
struct Model {

    flux_ftype<dtype, dimState> flux;                     // F: Q -> F(Q)
    waveSpeed_ftype<dtype, dimState> wave_speed;          // \rho( A(Q) )
    riemannSolver_ftype<dtype, dimState> riemann_solver;  // R(QL, QR, x/t)

    Model(flux_ftype<dtype, dimState> flux,
          waveSpeed_ftype<dtype, dimState> wave_speed,
          riemannSolver_ftype<dtype, dimState> riemann_solver
    ): flux(flux), wave_speed(wave_speed), riemann_solver(riemann_solver) {}
};

/*###########################################
#--------- Model Maker (abstract) --------- #
###########################################*/

/** @brief Produces 1D models*/
template < typename dtype, unsigned int dimState >
struct ModelMaker {
    virtual Model<dtype, dimState> make_normal_model(const std::array<double, 2>&) const = 0;
    virtual Model<dtype, dimState> make_normal_model() const = 0;
};

/*#################################
#--------- Linear Maker --------- #
#################################*/

/** @brief Produces 1D linear models*/
template < typename dtype, unsigned int dimState >
struct LinearMaker: ModelMaker<dtype, dimState> {

    EigType::Matrix<dtype, dimState> flux_x;
    EigType::Matrix<dtype, dimState> flux_y;

    /** @brief Private method to build models from matrix*/
    Model<dtype, dimState> model(const EigType::Matrix<dtype, dimState>& A_flux) const {
        // Build flux
        flux_ftype<dtype, dimState> flux = [A_flux](const EigType::Vector<dtype, dimState>& Q) {
            return A_flux*Q;
        };

        // Build wave speed
        EigType::EigenStructure<dtype, dimState> eig_struct = A_flux.get_eigen_structure();

        double max_wave_speed = eig_struct.eig_val.abs_max();
        waveSpeed_ftype<dtype, dimState> wave_speed = [max_wave_speed](const EigType::Vector<dtype, dimState>& Q) {
            return max_wave_speed;
        };

        // Build Riemann solver
        riemannSolver_ftype<dtype, dimState> riemann_solver = [eig_struct](const EigType::Vector<dtype, dimState>& QL, const EigType::Vector<dtype, dimState>& QR, double xi) {
            EigType::Vector<dtype, dimState> dQ(QR - QL);
            EigType::Vector<dtype, dimState> alpha(eig_struct.eig_inv*dQ);
            EigType::Vector<dtype, dimState> coeffs;
            for (index_t i = 0; i < dimState; ++i)
                coeffs(i) = (eig_struct.eig_val(i) < xi) ? alpha(i) : 0.0;
            return QL + eig_struct.eig_vec * QL;
        };

        return Model<dtype, dimState>(flux, wave_speed, riemann_solver);
    }

    public:
        LinearMaker(const EigType::Matrix<dtype, dimState>& flux_x,
                    const EigType::Matrix<dtype, dimState>& flux_y): flux_x(flux_x), flux_y(flux_y) {}
        LinearMaker(const EigType::Matrix<dtype, dimState>& flux_x): flux_x(flux_x) {}

        /** @brief Builds a regular 1D model*/
        Model<dtype, dimState> make_normal_model() const;
        /** @brief Builds a 1D projected 2D model*/
        Model<dtype, dimState> make_normal_model(const std::array<double, 2>& normal) const;
};

template < typename dtype, unsigned int dimState >
Model<dtype, dimState> LinearMaker<dtype, dimState>::make_normal_model() const {
    EigType::Matrix<dtype, dimState> A_flux(flux_x);
    return LinearMaker<dtype, dimState>::model(A_flux);
}

template < typename dtype, unsigned int dimState >
Model<dtype, dimState> LinearMaker<dtype, dimState>::make_normal_model(const std::array<double, 2>& normal) const {
    double nx = normal[0];
    double ny = normal[1];
    EigType::Matrix<dtype, dimState> A_flux(flux_x*nx + flux_y*ny);
    return LinearMaker<dtype, dimState>::model(A_flux);
}

/*######################################
#--------- Linear Maker (1D) --------- #
######################################*/

/** @brief Linear Maker specialization for 1D dimState.
 * Overall more efficient in the scalar case.
*/
template <typename dtype>
struct LinearMaker<dtype, 1>: ModelMaker<dtype, 1> {

    dtype flux_x;
    dtype flux_y;

    /** @brief Private method to build models from matrix*/
    Model<dtype, 1> model(dtype c) const {

        // Build flux
        flux_ftype<dtype, 1> flux = [c](const dtype& Q) {
            return Q*c;
        };

        // Build wave speed
        double max_wave_speed = std::abs( c );
        waveSpeed_ftype<dtype, 1> wave_speed = [max_wave_speed](const dtype& Q) {
            return max_wave_speed;
        };

        // Build Riemann solver
        riemannSolver_ftype<dtype, 1> riemann_solver = [c](const dtype& QL, const dtype& QR, dtype xi) {
            if (c < xi)
                return QR;
            return QR;
        };

        return Model<dtype, 1>(flux, wave_speed, riemann_solver);
    }

    public:
        LinearMaker(dtype flux_x, dtype flux_y): flux_x(flux_x), flux_y(flux_y) {}
        LinearMaker(dtype flux_x): flux_x(flux_x) {}

        /** @brief Builds a regular 1D model*/
        Model<dtype, 1> make_normal_model() const;
        /** @brief Builds a 1D projected 2D model*/
        Model<dtype, 1> make_normal_model(const std::array<double, 2>& normal) const;
};

template < typename dtype >
Model<dtype, 1> LinearMaker<dtype, 1>::make_normal_model() const {
    return model(flux_x);
}

template < typename dtype >
Model<dtype, 1> LinearMaker<dtype, 1>::make_normal_model(const std::array<double, 2>& normal) const {
    double nx = normal[0];
    double ny = normal[1];
    dtype c = flux_x*nx + flux_y*ny;
    return model(c);
}

#endif