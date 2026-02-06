#ifndef MODELS_HPP
#define MODELS_HPP

#include <iostream>
#include <algorithm>
#include <memory>
#include <cmath>

#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
using namespace FVTYPES;

/*#####################################
#--------- Model (abstract) --------- #
#####################################*/

/**@brief One dimensional conservation model*/
template < typename dtype, size_t dimState >
struct Model {

    virtual Var<dtype, dimState> flux( const Var<dtype, dimState>& ) const = 0;
    virtual dtype wave_speed( const Var<dtype, dimState>& ) const = 0;
    virtual Var<dtype, dimState> riemann_solver( const Var<dtype, dimState>&, const Var<dtype, dimState>&, dtype ) const = 0;

    Model() = default;
};

/*#################################
#--------- Linear Model --------- #
#################################*/

template < typename dtype, size_t dimState >
struct LinearModel: Model<dtype, dimState> {

    EigType::Matrix<dtype, dimState> A_flux;
    
    EigType::Matrix<dtype, dimState> eig_inv;
    EigType::Matrix<dtype, dimState> eig_vec;
    std::array<dtype, dimState> eig_val;

    dtype max_wave_speed;

    Var<dtype, dimState> flux( const Var<dtype, dimState>& Q ) const {return A_flux*Q;}
    dtype wave_speed( const Var<dtype, dimState>& Q ) const {return max_wave_speed;}
    Var<dtype, dimState> riemann_solver( const Var<dtype, dimState>& QL, const Var<dtype, dimState>& QR, dtype xi ) const {
        Var<dtype, dimState> dQ = QR - QL;
        Var<dtype, dimState> alpha = eig_inv * dQ;
        Var<dtype, dimState> coeffs;
        for (size_t i = 0; i < dimState; ++ i)
            coeffs[i] = (eig_val[i] < xi) ? alpha[i] : 0.0;
        return QL + eig_vec * QL;
    }

    LinearModel(const EigType::Matrix<dtype, dimState>& A_flux): A_flux(A_flux) {
        EigType::EigenStructure<dtype, dimState> eig_struct = A_flux.get_eigen_structure();
        eig_inv = eig_struct.eig_inv;
        eig_vec = eig_struct.eig_vec;
        eig_val = eig_struct.eig_val;

        max_wave_speed = std::abs(eig_val[0]);
        for (size_t i = 1; i < dimState; ++ i) {
            if (std::abs(eig_val[i]) > max_wave_speed)
                max_wave_speed = std::abs(eig_val[i]);
        }
    }
};

template < typename dtype >
struct LinearModel<dtype, 1>: Model<dtype, 1> {

    dtype c_speed;

    dtype flux( const dtype& Q ) const {return c_speed*Q;}
    dtype wave_speed( const dtype& Q ) const {return std::abs(c_speed);}
    dtype riemann_solver( const dtype& QL, const dtype& QR, dtype xi ) const {return (c_speed < xi) ? QR : QR;}

    LinearModel(const dtype& c_speed): c_speed(c_speed) {}
};

/*###########################################
#--------- Model Maker (abstract) --------- #
###########################################*/

/** @brief Produces 1D models*/
template < typename dtype, size_t dimState >
struct ModelMaker {
    
    virtual std::unique_ptr< Model<dtype, dimState> > make_normal_model() const = 0;
    virtual std::unique_ptr< Model<dtype, dimState> > make_normal_model(const std::array<double, 2>&) const = 0;

    ModelMaker() = default;
};

/*#################################
#--------- Linear Maker --------- #
#################################*/

/** @brief Produces 1D linear models*/
template < typename dtype, size_t dimState >
struct LinearMaker: ModelMaker<dtype, dimState> {

    EigType::Matrix<dtype, dimState> flux_x;
    EigType::Matrix<dtype, dimState> flux_y;

    std::unique_ptr< Model<dtype, dimState> > make_normal_model() const;
    std::unique_ptr< Model<dtype, dimState> > make_normal_model(const std::array<double, 2>& normal) const;

    LinearMaker(const EigType::Matrix<dtype, dimState>& flux_x): flux_x(flux_x) {}
    LinearMaker(const EigType::Matrix<dtype, dimState>& flux_x,
                const EigType::Matrix<dtype, dimState>& flux_y): flux_x(flux_x), flux_y(flux_y) {}
};

template < typename dtype, size_t dimState >
std::unique_ptr< Model<dtype, dimState> > LinearMaker<dtype, dimState>::make_normal_model() const {
    return std::make_unique< LinearModel<dtype, dimState> >(flux_x);
}

template < typename dtype, size_t dimState >
std::unique_ptr< Model<dtype, dimState> > LinearMaker<dtype, dimState>::make_normal_model(const std::array<double, 2>& normal) const {
    double nx = normal[0];
    double ny = normal[1];
    EigType::Matrix<dtype, dimState> A_flux(flux_x*nx + flux_y*ny);
    return std::make_unique< LinearModel<dtype, dimState> >(A_flux);
}

/*######################################
#--------- Linear Maker (1D) --------- #
######################################*/

/** @brief Produces 1D linear models*/
template < typename dtype >
struct LinearMaker<dtype, 1>: ModelMaker<dtype, 1> {

    dtype flux_x;
    dtype flux_y;

    std::unique_ptr< Model<dtype, 1> > make_normal_model() const;
    std::unique_ptr< Model<dtype, 1> > make_normal_model(const std::array<double, 2>& normal) const;

    LinearMaker(const dtype& flux_x): flux_x(flux_x) {}
    LinearMaker(const dtype& flux_x, const dtype& flux_y): flux_x(flux_x), flux_y(flux_y) {}
};

template < typename dtype >
std::unique_ptr< Model<dtype, 1> > LinearMaker<dtype, 1>::make_normal_model() const {
    return std::make_unique< LinearModel<dtype, 1> >(flux_x);
}

template < typename dtype >
std::unique_ptr< Model<dtype, 1> > LinearMaker<dtype, 1>::make_normal_model(const std::array<double, 2>& normal) const {
    double nx = normal[0];
    double ny = normal[1];
    dtype A_flux = flux_x*nx + flux_y*ny;
    return std::make_unique< LinearModel<dtype, 1> >(A_flux);
}

#endif