#ifndef MESH2D_HPP
#define MESH2D_HPP

#include <vector>
#include <array>

#include "utils/linear.hpp"
#include "utils/typdefs.hpp"
using namespace FVTYPES;

/*#############################
#--------- Field 2D --------- #
#############################*/

/** @brief 2D field of dimState data*/
template < unsigned int dimState >
class Field2D {
    
    private:
        // ATTRIBUTES
        const static int dim = dimState;
        size_t nx; size_t ny;
        std::vector< StateType::Var<dimState> > values;

    public:
        Field2D(size_t nx, size_t ny): nx(nx), ny(ny) {
            values.resize( (nx+2)*(ny+2) );
        }
        // ACCESSORS
        StateType::Var<dimState>& operator()(index_t i, index_t j) {return values[ (j%(ny+2))*(nx+2) + (i%(nx+2)) ];}
        const StateType::Var<dimState>& operator()(index_t i, index_t j) const {return values[ (j%(ny+2))*(nx+2) + (i%(nx+2)) ];}
};

/*############################
#--------- Mesh 2D --------- #
############################*/

/** @brief 2D Cartesian mesh*/
class Mesh2D {

    private:
        // ATTRIBUTES
        float x_length; float y_length;
        size_t nCx; size_t nPx;
        size_t nCy; size_t nPy;
        float dx; float dy;

        std::vector< xyPoint > cell_centers;
        std::vector< xyPoint > cell_corners;

    public:

        Mesh2D(float Lx, float Ly, size_t nx, size_t ny) {
            x_length = Lx; y_length = Ly;

            nCx = nx; nPx = nx+1;
            nCy = ny; nPy = ny+1;
            dx = Lx/nx; dy = Ly/ny;

            cell_centers.resize( (nCx+2)*(nCy+2) );
            cell_corners.resize( (nPx+2)*(nPy+2) );
            
            for (size_t j = 0; j < nCy+2; ++j) {
                for (size_t i = 0; i < nCx+2; ++i) {
                    xyPoint center = {-0.5*dx + i*dx, -0.5*dy + j*dy};
                    cell_centers[j*(nCx+2) + i] = center;            
                }
            }

            for (size_t j = 0; j < nPy+2; ++j) {
                for (size_t i = 0; i < nPx+2; ++ i) {
                    xyPoint corner = {-dx + j*dx, -dy + i*dy};
                    cell_corners[j*(nPx+2) + i] = corner;
                }
            }
        }
        // ACCESSORS
        float get_dx() const {return dx;}
        float get_dy() const {return dy;}
        size_t get_nCx() const {return nCx;}
        size_t get_nCy() const {return nCy;}
        size_t get_nPx() const {return nPx;}
        size_t get_nPy() const {return nPy;}

        // METHODS
        /** @brief Returns cell (i,j) <-> (x,y)*/
        xyPoint get_cell(int i, int j) const {
            return cell_centers[ (j%(nCy+2))*(nCx+2) + (i%(nCx+2)) ];
        }
        /** @brief Returns cell (i,j) <-> (x,y)*/
        xyPoint get_corner(int i, int j) const {
            return cell_corners[ (j%(nPy+2))*(nPx+2) + (i%(nPx+2)) ];
        }
        /** @brief  Returns cell-centered field*/
        template < unsigned int dimState > 
        Field2D<dimState> get_center_field() const {
            return Field2D<dimState>(nCx, nCy);
        }
        /** @brief Returns corner-centered field*/
        template < unsigned int dimState > 
        Field2D<dimState> get_corner_field() const {
            return Field2D<dimState>(nPx, nPy);
        }
        /** @brief Returns cell-evaluated field*/
        template < unsigned int dimState >
        Field2D<dimState> evaluate_center( const fn_xy_ftype<dimState>& f_init ) const {
            Field2D<dimState> field = get_center_field<dimState>();

            for (size_t j = 0; j < nCy+2; ++j) {
                for (size_t i = 0; i < nCx+2; ++i) {
                    field(i,j) = f_init(get_cell(i,j));
                }
            }

            return field;
        }
        /** @brief Returns corner-evaluated field*/
        template < unsigned int dimState >
        Field2D<dimState> evaluate_corner( const fn_xy_ftype<dimState>& f_init ) const {
            Field2D<dimState> field = get_corner_field<dimState>();

            for (size_t j = 0; j < nCy+3; ++j) {
                for (size_t i = 0; i < nCx+3; ++i) {
                    field(i,j) = f_init(get_corner(i,j));
                }
            }

            return field;
        }
};
#endif
