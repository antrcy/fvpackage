#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <iostream>
#include <cmath>
#include <array>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <complex>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>

namespace EigType 
{
    template < typename dtype, size_t dimState >
    struct Matrix;

    template < typename dtype, size_t dimState >
    struct EigenStructure;
}

/*##########################
#--------- Array --------- #
##########################*/

// Basic utility functions for std::array

template < typename dtype, size_t dimState >
std::array<dtype, dimState> operator+(
    const std::array<dtype, dimState>& a, 
    const std::array<dtype, dimState>& b ) 
{
    std::array<dtype, dimState> result;
    for (size_t i = 0; i < dimState; ++ i)
        result[i] = a[i] + b[i];
    return result;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState>& operator+=(
    std::array<dtype, dimState>& a, 
    const std::array<dtype, dimState>& b ) 
{
    for (size_t i = 0; i < dimState; ++ i)
        a[i] += b[i];
    return a;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState> operator-(
    const std::array<dtype, dimState>& a, 
    const std::array<dtype, dimState>& b )
{
    std::array<dtype, dimState> result;
    for (size_t i = 0; i < dimState; ++ i)
        result[i] = a[i] - b[i];
    return result;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState>& operator-=(
    std::array<dtype, dimState>& a, 
    const std::array<dtype, dimState>& b ) 
{
    for (size_t i = 0; i < dimState; ++ i)
        a[i] -= b[i];
    return a;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState> operator*(
    const std::array<dtype, dimState>& a, const dtype& l )
{
    std::array<dtype, dimState> result;
    for (size_t i = 0; i < dimState; ++ i)
        result[i] = a[i]*l;
    return result;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState>& operator*=(
    std::array<dtype, dimState>& a, const dtype& l )
{
    for (size_t i = 0; i < dimState; ++ i)
        a[i] *= l;
    return a;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState> operator*(
    const std::array<dtype, dimState>& a, const float& l )
{
    std::array<dtype, dimState> result;
    for (size_t i = 0; i < dimState; ++ i)
        result[i] = a[i]*l;
    return result;
}

template < typename dtype, size_t dimState >
std::array<dtype, dimState>& operator*=(
    std::array<dtype, dimState>& a, const float& l )
{
    for (size_t i = 0; i < dimState; ++ i)
        a[i] *= l;
    return a;
}

/** @brief Sorts vec in place and returns the permutation*/
template < typename dtype, size_t dimState >
std::array<size_t, dimState> bubble_sort( std::array<dtype, dimState>& vec ) {
    std::array<size_t, dimState> args;
    for (size_t i = 0; i < dimState; ++ i) {
        args[i] = i;
    }

    bool swapped;
    for (size_t i = 0; i < dimState - 1; ++ i) {
        swapped = false;
        for (size_t j = 0; j < dimState - i - 1; ++ j) {
            if (vec[j] > vec[j + 1]) {
                std::swap(vec[j], vec[j + 1]);
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
template < typename dtype, size_t dimState >
struct EigType::EigenStructure {
    std::array<dtype, dimState> eig_val;
    EigType::Matrix<dtype, dimState> eig_vec;
    EigType::Matrix<dtype, dimState> eig_inv;
};

/*###########################
#--------- Matrix --------- #
###########################*/

/** @brief Square matrix wrapper for Eigen::Matrix.*/
template < typename dtype, size_t dimState >
struct  EigType::Matrix {
    // ATTRIBUTES
    Eigen::Matrix<dtype, dimState, dimState> mat;

    // METHODES
    Matrix() = default;
    Matrix(const Eigen::Matrix<dtype, dimState, dimState>& mat): mat( mat ) {}
    Matrix(const std::initializer_list< std::initializer_list<dtype> >& init_list): 
        mat( Eigen::Matrix<dtype, dimState, dimState>(init_list) ) {}
    explicit Matrix(dtype init) {
        static_assert(dimState == 1);
        mat(0,0) = init;
    }
    
    Matrix& operator=(const Matrix&) = default;
    dtype& operator()(size_t i, size_t j) {return mat(i,j);}
    const dtype& operator()(size_t i, size_t j) const {return mat(i,j);}

    /** @brief Operator overload 'Matrix + Matrix' */
    Matrix<dtype, dimState> operator+(const Matrix<dtype, dimState>& m) {
        EigType::Matrix<dtype, dimState> res; res.mat = mat + m.mat;
        return res;
    }
    /** @brief Operator overload 'Matrix * double'*/
    Matrix<dtype, dimState> operator*(dtype l) const {
        EigType::Matrix<dtype, dimState> res; res.mat = mat*l;
        return res;
    }
    /** @brief Operator overload 'Matrix * Vector'*/
    std::array<dtype, dimState> operator*( const std::array<dtype, dimState>& vec ) const {
        std::array<dtype, dimState> res;
        for (size_t i = 0; i < dimState; ++ i) {
            res[i] = 0;
            for (size_t j = 0; j < dimState; ++ j)
                res[i] += mat(i,j)*vec[j];
        }
        return res;
    }

    Matrix<dtype, dimState> permute_col( std::array<size_t, dimState> perm );
    EigenStructure<dtype, dimState> get_eigen_structure() const;
};

/** @brief Permutes columns according to pattern perm*/
template < typename dtype, size_t dimState >
EigType::Matrix<dtype, dimState> EigType::Matrix<dtype, dimState>::permute_col( std::array<size_t, dimState> perm ) {
    EigType::Matrix<dtype, dimState> res;

    for (size_t i = 0; i < dimState; ++ i) {
        res.mat.col(i) = mat.col(perm[i]);
    }

    return res;
}

/** @brief Computes the eigen-structure of the matrix*/
template < typename dtype, size_t dimState >
EigType::EigenStructure<dtype, dimState> EigType::Matrix<dtype, dimState>::get_eigen_structure() const {
    Eigen::EigenSolver< Eigen::Matrix<dtype, dimState, dimState> > es(mat);

    // Compute real eigenvalues and sort them
    std::array<dtype, dimState> eig_val;  
    Eigen::Vector<dtype, dimState> eig_res = es.eigenvalues().real();
    for (size_t i = 0; i < dimState; i ++) {
        eig_val[i] = eig_res(i);
    }

    std::array<size_t, dimState> perm = bubble_sort(eig_val);

    // Sort eigenvectors according to the same order
    EigType::Matrix<dtype, dimState> eig_vec( es.eigenvectors().real() );
    eig_vec.permute_col(perm);

    // Compute inverse eigen matrix
    EigType::Matrix<dtype, dimState> eig_inv( eig_vec.mat.inverse() );

    EigType::EigenStructure<dtype, dimState> eig_struct;
    eig_struct.eig_val = eig_val;
    eig_struct.eig_vec = eig_vec;
    eig_struct.eig_inv = eig_inv;

    return eig_struct;
}

#endif