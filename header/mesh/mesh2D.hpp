#ifndef MESH2D_HPP
#define MESH2D_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
#include <string>

#include "utils/typdefs.hpp"
#include "utils/linear.hpp"
using namespace FVTYPES;

class Mesh2D;

/*#############################
#--------- Field 2D --------- #
#############################*/

/** @brief 2D field of dimState data*/
template < typename dtype, size_t dimState >
class Field2D {
    private:
        // ATTRIBUTES
        size_t nx; size_t ny;
        std::vector< Var<dtype, dimState> > values;

    public:
        Field2D(size_t nx, size_t ny): nx(nx), ny(ny) {values.resize( (nx+2)*(ny+2) );}
        Field2D(const Field2D<dtype, dimState>& field): nx(field.nx), ny(field.ny) {
            values.reserve( (nx+2)*(ny+2) );
            for (const auto& val: field.values) {
                values.push_back(val);
            }
        }

        // ACCESSORS
        Var<dtype, dimState>& operator()(size_t i, size_t j) {return values[ j*(nx+2) + i ];}
        const Var<dtype, dimState>& operator()(size_t i, size_t j) const {return values[ j*(nx+2) + i ];}
        size_t get_nx() const {return nx;}
        size_t get_ny() const {return ny;}

        Field2D<dtype, dimState> operator+(const Field2D<dtype, dimState>& field) const {
            Field2D<dtype, dimState> result( *this );
            for (size_t i = 0; i < values.size(); ++ i) {
                result.values[i] += field.values[i];
            } return result;
        }

        Field2D<dtype, dimState> operator*(const dtype& l) const {
            Field2D<dtype, dimState> result( *this );
            for (size_t i = 0; i < values.size(); ++ i) {
                result.values[i] *= l;
            } return result;
        }

        Field2D<dtype, dimState>& operator+=(const Field2D<dtype, dimState>& field) {
            for (size_t i = 0; i < values.size(); ++ i) {
                values[i] += field.values[i];
            } return *this;
        }

        Field2D<dtype, dimState>& operator*=(const dtype& l) const {
            for (size_t i = 0; i < values.size(); ++ i) {
                values[i] *= l;
            } return *this;
        }

        void save_to_csv(const Mesh2D&, std::string) const;
};

/** @brief 2D field of dimState data*/
template < typename dtype >
class Field2D<dtype, 1> {
    private:
        // ATTRIBUTES
        size_t nx; size_t ny;
        std::vector< dtype > values;

    public:
        Field2D(size_t nx, size_t ny): nx(nx), ny(ny) {
            values.resize( (nx+2)*(ny+2) );
        }
        // ACCESSORS
        dtype& operator()(size_t i, size_t j) {return values[ j*(nx+2) + i ];}
        const dtype& operator()(size_t i, size_t j) const {return values[ j*(nx+2) + i ];}
        size_t get_nx() const {return nx;}
        size_t get_ny() const {return ny;}

        void save_to_csv(const Mesh2D&, std::string) const;
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
        Mesh2D(float Lx, float Ly, size_t nx, size_t ny): 
               x_length(Lx), y_length(Ly), 
               nCx(nx), nCy(ny), nPx(nx+1), nPy(ny+1),
               dx(Lx/nx), dy(Ly/ny)
         {

            cell_centers.resize( (nCx+2)*(nCy+2) );
            cell_corners.resize( (nPx+2)*(nPy+2) );
            
            for (size_t j = 0; j < nCy+2; ++ j) {
                for (size_t i = 0; i < nCx+2; ++ i) {
                    xyPoint center = {-0.5*dx + i*dx, -0.5*dy + j*dy};
                    cell_centers[j*(nCx+2) + i] = center;            
                }
            }

            for (size_t j = 0; j < nPy+2; ++ j) {
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
            return cell_centers[ j*(nCx+2) + i ];
        }
        /** @brief Returns cell (i,j) <-> (x,y)*/
        xyPoint get_corner(int i, int j) const {
            return cell_corners[ j*(nPx+2) + i ];
        }
        /** @brief  Returns cell-centered field*/
        template < typename dtype, size_t dimState > 
        Field2D<dtype, dimState> get_center_field() const {
            return Field2D<dtype, dimState>(nCx, nCy);
        }
        /** @brief Returns corner-centered field*/
        template < typename dtype, size_t dimState > 
        Field2D<dtype, dimState> get_corner_field() const {
            return Field2D<dtype, dimState>(nPx, nPy);
        }
        /** @brief Returns cell-evaluated field*/
        template < typename dtype, size_t dimState >
        Field2D<dtype, dimState> evaluate_center( const fnXY_ftype<dtype, dimState>& f_init ) const {
            Field2D<dtype, dimState> field = get_center_field<dtype, dimState>();

            for (size_t j = 0; j < nCy+2; ++ j) {
                for (size_t i = 0; i < nCx+2; ++ i) {
                    field(i,j) = f_init(get_cell(i,j));
                }
            }

            return field;
        }
        /** @brief Returns corner-evaluated field*/
        template < typename dtype, size_t dimState >
        Field2D<dtype, dimState> evaluate_corner( const fnXY_ftype<dtype, dimState>& f_init ) const {
            Field2D<dtype, dimState> field = get_corner_field<dtype, dimState>();

            for (size_t j = 0; j < nCy+3; ++ j) {
                for (size_t i = 0; i < nCx+3; ++ i) {
                    field(i,j) = f_init(get_corner(i,j));
                }
            }

            return field;
        }
};

template < typename dtype, size_t dimState >
void Field2D<dtype, dimState>::save_to_csv(const Mesh2D& mesh, std::string file_name) const {
    std::ofstream file(file_name);

    std::cout << "writing to " << file_name << std::endl;
    
    file << "x,y";
    for (size_t k = 0; k < dimState; ++ k) file << ",v" << k;
    for (size_t i = 0; i < nx+2; ++ i) {
        for ( size_t j = 0; j < ny+2; ++ j) {
            xyPoint p = mesh.get_cell(i,j);
            file << "\n" << p[0] << ',' << p[1];
            for (size_t k = 0; k < dimState; ++ k) {
                file << ',' << this->operator()(i,j)[k];
            }
        }
    }

    file.close();
}

template < typename dtype >
void Field2D<dtype, 1>::save_to_csv(const Mesh2D& mesh, std::string file_name) const {
    std::ofstream file(file_name);

    file << "x,y,v";
    for (size_t i = 0; i < nx+2; ++ i) {
        for (size_t j = 0; j < ny+2; ++ j) {
            xyPoint p = mesh.get_cell(i,j);
            file << '\n' << p[0] << ',' << p[1];
            file << this->operator()(i,j);
        }
    }

    file.close();
}


#endif