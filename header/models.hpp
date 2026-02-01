#ifndef MODELS_HPP
#define MODELS_HPP

#include <iostream>
#include <functional>
#include <algorithm>
#include <cmath>

#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
using namespace FVTYPES;

/*##########################
#--------- Model --------- #
##########################*/

/** @brief One dimensional conservation model*/
template < unsigned int dimState >
struct Model {
    // ATTRIBUTES
    static const unsigned int dim = dimState;

    flux_ftype<dimState> flux;                      // F: Q -> F(Q)
    wave_speed_ftype<dimState> wave_speed;          // \rho( A(Q) )
    riemann_solver_ftype<dimState> riemann_solver;  // R(QL, QR, x/t)

    Model(flux_ftype<dimState> flux,
          wave_speed_ftype<dimState> wave_speed,
          riemann_solver_ftype<dimState> riemann_solver
    ): flux(flux), wave_speed(wave_speed), riemann_solver(riemann_solver) {}
};

/*################################
#--------- Model Maker --------- #
################################*/

/** @brief Produces 1D models*/
template < unsigned int dimState >
class ModelMaker {
    public:
        virtual Model<dimState> make_normal_model(const std::array<double, 2>&) const = 0;
};

/*#################################
#--------- Linear Maker --------- #
#################################*/

/** @brief Produces 1D linear models*/
template < unsigned int dimState >
class LinearMaker: ModelMaker<dimState> {
    private:
        // ATTRIBUTES
        static const unsigned int dim = dimState;

        EigType::Matrix<dimState> flux_x;
        EigType::Matrix<dimState> flux_y;

        /** @brief Private method to build models from matrix*/
        Model<dimState> model(const EigType::Matrix<dimState>& A_flux) const {
            // Build flux
            flux_ftype<dimState> flux = [A_flux](const EigType::Vector<dimState>& Q) {
                return A_flux*Q;
            };

            // Build wave speed
            EigType::EigenStructure<dimState> eig_struct = A_flux.get_eigen_structure();

            double max_wave_speed = eig_struct.eig_val.abs_max();
            wave_speed_ftype<dimState> wave_speed = [max_wave_speed](const EigType::Vector<dimState>& Q) {
                return max_wave_speed;
            };

            // Build Riemann solver
            riemann_solver_ftype<dimState> riemann_solver = [eig_struct](const EigType::Vector<dimState>& QL, const EigType::Vector<dimState>& QR, double xi) {
                EigType::Vector<dimState> dQ(QR - QL);
                EigType::Vector<dimState> alpha(eig_struct.eig_inv*dQ);
                EigType::Vector<dimState> coeffs;
                for (index_t i = 0; i < dimState; ++i)
                    coeffs(i) = (eig_struct.eig_val(i) < xi) ? alpha(i) : 0.0;
                return QL + eig_struct.eig_vec * QL;
            };

            return Model<dimState>(flux, wave_speed, riemann_solver);
        }

    public:
        LinearMaker(const EigType::Matrix<dimState>& flux_x,
                    const EigType::Matrix<dimState>& flux_y): flux_x(flux_x), flux_y(flux_y) {}
        LinearMaker(const EigType::Matrix<dimState>& flux_x): flux_x(flux_x) {}
        // METHODS
        /** @brief Builds a regular 1D model*/
        Model<dimState> make_model() const;
        /** @brief Builds a 1D projected 2D model*/
        Model<dimState> make_normal_model(const std::array<double, 2>& normal) const;
};

template < unsigned int dimState >
Model<dimState> LinearMaker<dimState>::make_model() const {
    EigType::Matrix<dimState> A_flux(flux_x);
    return LinearMaker<dimState>::model(A_flux);
}

template < unsigned int dimState >
Model<dimState> LinearMaker<dimState>::make_normal_model(const std::array<double, 2>& normal) const {
    double nx = normal[0];
    double ny = normal[1];
    EigType::Matrix<dimState> A_flux(flux_x*nx + flux_y*ny);
    return LinearMaker<dimState>::model(A_flux);
}

/*######################################
#--------- Linear Maker (1D) --------- #
######################################*/

/** @brief Linear Maker specialization for 1D dimState.
 * Overall more efficient in the scalar case.
*/
template <>
class LinearMaker<1>: ModelMaker<1> {
    private:
        // ATTRIBUTES
        static const unsigned int dim = 1;

        double flux_x;
        double flux_y;

        /** @brief Private method to build models from matrix*/
        Model<1> model(double c) const {

            // Build flux
            flux_ftype<1> flux = [c](const double& Q) {
                return Q*c;
            };

            // Build wave speed
            double max_wave_speed = std::abs( c );
            wave_speed_ftype<1> wave_speed = [max_wave_speed](const double& Q) {
                return max_wave_speed;
            };

            // Build Riemann solver
            riemann_solver_ftype<1> riemann_solver = [c](const double& QL, const double& QR, double xi) {
                if (c < xi)
                    return QR;
                return QR;
            };

            return Model<1>(flux, wave_speed, riemann_solver);
        }

    public:
        LinearMaker(double flux_x, double flux_y): flux_x(flux_x), flux_y(flux_y) {}
        LinearMaker(double flux_x): flux_x(flux_x) {}
        // METHODS
        /** @brief Builds a regular 1D model*/
        Model<1> make_model() const;
        /** @brief Builds a 1D projected 2D model*/
        Model<1> make_normal_model(const std::array<double, 2>& normal) const;
};

Model<1> LinearMaker<1>::make_model() const {
    return model(flux_x);
}

Model<1> LinearMaker<1>::make_normal_model(const std::array<double, 2>& normal) const {
    double nx = normal[0];
    double ny = normal[1];
    double c = flux_x*nx + flux_y*ny;
    return model(c);
}

#endif