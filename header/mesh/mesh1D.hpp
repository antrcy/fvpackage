#ifndef MESH1D_HPP
#define MESH1D_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <eigen3/Eigen/src/Core/util/Memory.h>

#include "utils/linear.hpp"
#include "utils/typdefs.hpp"
using namespace FVTYPES;

/*#############################
#--------- Field 1D --------- #
#############################*/

/** @brief 1D field of dimState data*/
template < unsigned int dimState >
class Field1D {
    private:
        // ATTRIBUTES
        const static int dim = dimState;
        size_t nx;
        std::vector< EigType::Vector<dimState>,
                     Eigen::aligned_allocator< EigType::Vector<dimState> > > values;

    public:
        Field1D(size_t nx): nx(nx) {
            values.resize( (nx+2) );
        }
        // ACCESSORS
        EigType::Vector<dimState>& operator()(index_t i) {return values[ i%(nx+2) ];}
        const EigType::Vector<dimState>& operator()(index_t i) const {return values[ i%(nx+2) ];}
        size_t get_nx() const {return nx;}
};

template <>
class Field1D<1> {
    private:
        // ATTRIBUTES
        size_t nx;
        std::vector< double > values;

    public:
        Field1D(size_t nx): nx(nx) {
            values.resize( (nx+2) );
        }
        // ACCESSORS
        double& operator()(index_t i) {return values[ i%(nx+2) ];}
        const double& operator()(index_t i) const {return values[ i%(nx+2) ];}
        size_t get_nx() const {return nx;}
};



/*############################
#--------- Mesh 1D --------- #
############################*/

/** @brief 1D Cartesian mesh*/
class Mesh1D {
    
    private:
        // ATTRIBUTES
        float x_length;
        size_t nCx; size_t nPx;
        float dx;

        std::vector< double > cell_centers;
        std::vector< double > cell_corners;

    public:

        Mesh1D(float Lx, size_t nx) {
            x_length = Lx;

            nCx = nx; nPx = nx+1;
            dx = Lx/nx;

            cell_centers.resize( (nCx+2) );
            cell_corners.resize( (nPx+2) );
            
            for (size_t i = 0; i < nCx+2; ++i) {
                double center = -0.5*dx + i*dx;
                cell_centers[i] = center;            
            }

            for (size_t i = 0; i < nPx+2; ++ i) {
                double corner = -dx + i*dx;
                cell_corners[i] = corner;
            }
        }
        // ACCESSORS
        float get_dx() const {return dx;}
        size_t get_nCx() const {return nCx;}
        size_t get_nPx() const {return nPx;}

        // METHODS
        /** @brief Returns cell (i) <-> (x)*/
        double get_cell(int i) const {
            return cell_centers[i];
        }
        /** @brief Returns corner (i) <-> (x)*/
        double get_corner(int i) const {
            return cell_corners[i];
        }
        /** @brief  Returns cell-centered field*/
        template < unsigned int dimState > 
        Field1D<dimState> get_center_field() const {
            return Field1D<dimState>(nCx);
        }
        /** @brief Returns corner-centered field*/
        template < unsigned int dimState > 
        Field1D<dimState> get_corner_field() const {
            return Field1D<dimState>(nPx);
        }
        /** @brief Returns cell-evaluated field*/
        template < unsigned int dimState >
        Field1D<dimState> evaluate_center( const fn_x_ftype<dimState>& f_init ) const {
            Field1D<dimState> field = get_center_field<dimState>();

            for (size_t i = 0; i < nCx+2; ++i) {
                field(i) = f_init(get_cell(i));
            }

            return field;
        }
        /** @brief Returns corner-evaluated field*/
        template < unsigned int dimState >
        Field1D<dimState> evaluate_corner( const fn_x_ftype<dimState>& f_init ) const {
            Field1D<dimState> field = get_corner_field<dimState>();

            for (size_t i = 0; i < nCx+3; ++i) {
                field(i) = f_init(get_corner(i));
            }

            return field;
        }
};

#endif