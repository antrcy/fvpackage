#include <iostream>
#include <tuple>
#include <eigen3/Eigen/Dense>

#include "mesh/mesh1D.hpp"
#include "utils/linear.hpp"
#include "utils/typdefs.hpp"
#include "models.hpp"
#include "fluxes.hpp"
#include "solver.hpp"

using namespace FVTYPES;

int main() {
    const unsigned int dimState = 1;
    using dtype = double;

    // Define mesh and initial condition
    Mesh1D mesh(1.0, 4000);
    fnX_ftype<dtype, dimState> u_init = [](double x) {
        return 1.0;
    };

    // Define a flux and a linear model
    dtype A = 1.0;
    //Eigen::Matrix<dtype, dimState, dimState> data({ {1.0, 0.0}, {0.0, 1.0} });
    //EigType::Matrix<dtype, dimState> A(data);
    LinearMaker<dtype, dimState> model_maker( A );

    // Define a flux maker and a solver
    RusanovFlux<dtype, dimState> rusanov;
    FiniteVolumeSolver<dtype, dimState> solver(rusanov);

    // Initialize fields and get solve step
    auto fields = solver.initialize(mesh, u_init);
    Field1D<dtype, dimState> Q1 = std::get<0>(fields);
    Field1D<dtype, dimState> Q2 = std::get<1>(fields);
    solveStep1D_ftype<dtype, dimState> solve_step = solver.get_solve_step(model_maker, mesh);
    
    // Time loop
    float time = 0; float T_final = 0.75; float dt = 0.5*mesh.get_dx();

    while (time < T_final) {
        solve_step(Q1, Q2, dt);
        std::swap(Q1, Q2);
        time += dt;
    }
}
