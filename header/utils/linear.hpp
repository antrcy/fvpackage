#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <iostream>
#include <cmath>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <complex>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/src/Core/util/Memory.h>

using index_t = unsigned long int;

namespace EigType {
    template < unsigned int dimState >
    struct Matrix;

    template < unsigned int dimState >
    struct Vector;

    template < unsigned int dimState >
    struct EigenStructure;
}

// Should be used with stateDim
namespace StateType {
    template < unsigned int dimState >
    using Var = std::conditional_t< dimState == 1, double, EigType::Vector<dimState> >;
}

/*###########################
#--------- Vector --------- #
###########################*/

/** @brief Vector wrapper for Eigen::Vector.*/
template < unsigned int dimState >
struct EigType::Vector {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // ATTRIBUTES

    Eigen::Vector<double, dimState> vec;

    // METHODES
    Vector() = default;
    Vector(const Vector&) = default;
    Vector(std::array<double, dimState> data) {
        for (index_t i = 0; i < dimState; ++i) {
            vec(i) = data[i];
        }
    }

    Vector& operator=(const Vector&) = default;

    double& operator()(index_t i) {return vec(i);}
    const double& operator()(index_t i) const {return vec(i);}

    /** @brief Operator overload 'Vector * double'*/
    Vector<dimState> operator*(double l) const {
        Vector<dimState> res; res.vec = vec*l;
        return res;
    }
    /** @brief Operator overload 'Vector + Vector'*/
    Vector<dimState> operator+(const Vector<dimState>& v) const {
        Vector<dimState> res; res.vec = vec + v.vec;
        return res;
    }
    /** @brief Operator overload 'Vector - Vector'*/
    Vector<dimState> operator-(const Vector<dimState>& v) const {
        Vector<dimState> res; res.vec = vec - v.vec;
        return res;
    }

    /** @brief Sorts eigenvectors and returns pattern.*/
    std::array<index_t, dimState> sort();

    /** @brief Computes highest absolute value eigenvalue.*/
    double abs_max() const;
};

template < unsigned int dimState >
double EigType::Vector<dimState>::abs_max() const {
    double max = 0.0;
    for (index_t i = 0; i < dimState; ++ i) {
        if ( std::abs( vec(i) )> max ) {
            max = std::abs( vec(i) );
        }
    }
    return max;
}

template < unsigned int dimState >
std::array<index_t, dimState> EigType::Vector<dimState>::sort() {
    std::array<index_t, dimState> args;
    for (index_t i = 0; i < dimState; ++ i) {
        args[i] = i;
    }

    bool swapped; // Buble sort 
    for (index_t i = 0; i < dimState - 1; i++) {
        swapped = false;
        for (index_t j = 0; j < dimState - i - 1; j++) {
            if (vec(j) > vec(j + 1)) {
                std::swap(vec(j), vec(j + 1));
                std::swap(args[j], args[j+1]);
                swapped = true;
            }
        }
    
        if (!swapped)
            break;
    }

    return args;
}

/*####################################
#--------- Eigen Structure --------- #
####################################*/

/** @brief Container for eigen-sructures.*/
template < unsigned int dimState >
struct EigType::EigenStructure {
    EigType::Vector<dimState> eig_val;
    EigType::Matrix<dimState> eig_vec;
    EigType::Matrix<dimState> eig_inv;
};

/*###########################
#--------- Matrix --------- #
###########################*/

/** @brief Square matrix wrapper for Eigen::Matrix.*/
template < unsigned int dimState >
struct  EigType::Matrix {
    // ATTRIBUTES

    Eigen::Matrix<double, dimState, dimState> mat;

    // METHODES
    Matrix() = default;
    Matrix(std::initializer_list<std::initializer_list<double>> init) 
        {mat = Eigen::Matrix<double, dimState, dimState>(init);}
    explicit Matrix(double init) {
        static_assert(dimState == 1);
        mat(0,0) = init;
    }

    /** @brief Set matrix data to data.*/
    void set(double** data) {
        for (index_t i = 0; i < dimState; ++i) {
            for (index_t j = 0; j < dimState; ++j) {
                mat(i,j) = data[i][j];
            }
        }
    }
    
    double& operator()(index_t i, index_t j) {return mat(i,j);}
    const double& operator()(index_t i, index_t j) const {return mat(i,j);}

    /** @brief Operator overload 'Matrix = Matrix'*/
    Matrix<dimState>& operator=(const Matrix<dimState>& m) {
        for (index_t i = 0; i < dimState; ++i) {
            for (index_t j = 0; j < dimState; ++j) {
                mat(i,j) = m.mat(i,j);
            }
        }
        return *this;
    }
    /** @brief Operator overload 'Matrix + Matrix' */
    Matrix<dimState> operator+(const Matrix<dimState>& m) {
        Matrix<dimState> res; res.mat = mat + m.mat;
        return res;
    }
    /** @brief Operator overload 'Matrix * double'*/
    Matrix<dimState> operator*(double l) const {
        Matrix<dimState> res; res.mat = mat*l;
        return res;
    }
    /** @brief Operator overload 'Matrix * Vector'*/
    Vector<dimState> operator*(const Vector<dimState>& vector) const {
        Vector<dimState> res;
        res.vec = mat*vector.vec;
        return res;
    }

    Matrix<dimState> permute_col(std::array<index_t, dimState> perm);
    EigenStructure<dimState> get_eigen_structure() const;
};

/** @brief Permutes columns according to pattern perm*/
template < unsigned int dimState >
EigType::Matrix<dimState> EigType::Matrix<dimState>::permute_col(std::array<index_t, dimState> perm) {
    Matrix<dimState> res;

    for (index_t i=0; i<dimState; ++i) {
        res.mat.col(i) = mat.col(perm[i]);
    }

    return res;
}

/** @brief Computes the eigen-structure of the matrix*/
template < unsigned int dimState >
EigType::EigenStructure<dimState> EigType::Matrix<dimState>::get_eigen_structure() const {
    Eigen::EigenSolver< Eigen::Matrix<double, dimState, dimState> > es(mat);

    // Compute real eigenvalues and sort them
    Vector<dimState> eig_val; eig_val.vec = es.eigenvalues().real();
    std::array<index_t, dimState> perm = eig_val.sort();

    // Sort eigenvectors according to the same order
    Matrix<dimState> eig_vec; eig_vec.mat = es.eigenvectors().real();
    eig_vec.permute_col(perm);

    // Compute inverse eigen matrix
    Matrix<dimState> eig_inv; eig_inv.mat = eig_vec.mat.inverse();

    EigenStructure<dimState> eig_struct;
    eig_struct.eig_val = eig_val;
    eig_struct.eig_vec = eig_vec;
    eig_struct.eig_inv = eig_inv;

    return eig_struct;
}

#endif