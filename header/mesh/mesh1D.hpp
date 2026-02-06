#ifndef MESH1D_HPP
#define MESH1D_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
#include <string>

#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
using namespace FVTYPES;

#ifndef IMAGE_PATH
#define IMAGE_PATH "/home/antoine/Internship/fvpackage/results"
#endif

class Mesh1D;

/*#############################
#--------- Field 1D --------- #
#############################*/

/** @brief 1D field of dimState data*/
template < typename dtype, size_t dimState >
class Field1D {
    private:
        // ATTRIBUTES
        size_t nx;
        std::vector< Var<dtype, dimState> > values;

    public:
        Field1D(size_t nx): nx(nx) {values.resize( (nx+2) );}

        // ACCESSORS
        Var<dtype, dimState>& operator()(size_t i) {return values[ i ];}
        const Var<dtype, dimState>& operator()(size_t i) const {return values[ i ];}
        size_t get_nx() const {return nx;}

        void save_to_csv(const Mesh1D&, std::string) const;
};

template < typename dtype >
class Field1D<dtype, 1> {
    private:
        // ATTRIBUTES
        size_t nx;
        std::vector< dtype > values;

    public:
        Field1D(size_t nx): nx(nx) {
            values.resize( (nx+2) );
        }
        // ACCESSORS
        dtype& operator()(size_t i) {return values[ i ];}
        const dtype& operator()(size_t i) const {return values[ i ];}
        size_t get_nx() const {return nx;}

        void save_to_csv(const Mesh1D&, std::string) const;
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

        std::vector< xPoint > cell_centers;
        std::vector< xPoint > cell_corners;

    public:
        Mesh1D(float Lx, size_t nx): x_length(Lx), nCx(nx), nPx(nx+1), dx(Lx/nx) {

            cell_centers.resize( (nCx+2) );
            cell_corners.resize( (nPx+2) );
            
            for (size_t i = 0; i < nCx+2; ++ i) {
                xPoint center = -0.5*dx + i*dx;
                cell_centers[i] = center;            
            }

            for (size_t i = 0; i < nPx+2; ++ i) {
                xPoint corner = -dx + i*dx;
                cell_corners[i] = corner;
            }
        }
        // ACCESSORS
        float get_dx() const {return dx;}
        size_t get_nCx() const {return nCx;}
        size_t get_nPx() const {return nPx;}

        // METHODS
        /** @brief Returns cell (i) <-> (x)*/
        xPoint get_cell(int i) const {
            return cell_centers[i];
        }
        /** @brief Returns corner (i) <-> (x)*/
        xPoint get_corner(int i) const {
            return cell_corners[i];
        }
        /** @brief  Returns cell-centered field*/
        template < typename dtype, size_t dimState > 
        Field1D<dtype, dimState> get_center_field() const {
            return Field1D<dtype, dimState>(nCx);
        }
        /** @brief Returns corner-centered field*/
        template < typename dtype, size_t dimState > 
        Field1D<dtype, dimState> get_corner_field() const {
            return Field1D<dtype, dimState>(nPx);
        }
        /** @brief Returns cell-evaluated field*/
        template < typename dtype, size_t dimState >
        Field1D<dtype, dimState> evaluate_center( const fnX_ftype<dtype, dimState>& f_init ) const {
            Field1D<dtype, dimState> field = get_center_field<dtype, dimState>();

            for (size_t i = 0; i < nCx+2; ++ i) {
                field(i) = f_init(get_cell(i));
            }

            return field;
        }
        /** @brief Returns corner-evaluated field*/
        template < typename dtype, size_t dimState >
        Field1D<dtype, dimState> evaluate_corner( const fnX_ftype<dtype, dimState>& f_init ) const {
            Field1D<dtype, dimState> field = get_corner_field<dtype, dimState>();

            for (size_t i = 0; i < nCx+3; ++ i) {
                field(i) = f_init(get_corner(i));
            }

            return field;
        }
};

template < typename dtype, size_t dimState >
void Field1D<dtype, dimState>::save_to_csv(const Mesh1D& mesh, std::string file_name) const {
    std::ofstream file(IMAGE_PATH+'/'+file_name);

    file << "x";
    for (size_t k = 0; k < dimState; ++ k) file << ",v" << k;
    for (size_t i = 0; i < nx+2; ++i) {
        file << '\n' << mesh.get_cell(i);
        for (size_t k = 0; k < dimState; ++ k) file << ',' << values[i](k);
    }

    file.close();
}

template < typename dtype >
void Field1D<dtype, 1>::save_to_csv(const Mesh1D& mesh, std::string file_name) const {
    std::ofstream file(IMAGE_PATH+'/'+file_name);

    file << "x,v";
    for (size_t i = 0; i < nx+2; ++ i)
        file << '\n' << mesh.get_cell(i) << ',' << values[i];
    
    file.close();
}

#endif