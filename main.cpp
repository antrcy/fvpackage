#include <iostream>
#include <tuple>
#include <eigen3/Eigen/Dense>

#include "mesh/mesh1D.hpp"
#include "utils/linear.hpp"
#include "utils/typdefs.hpp"
#include "models.hpp"
#include "solver.hpp"
using namespace FVTYPES;

int main() {
    const unsigned int dimState = 1;
    using dtype = double;

    // Initial condition
    fnX_ftype<dtype, dimState> u_init = [](double x) {
        return 1.0;
    };

    // Define model and solver
    Mesh1D mesh(1.0, 4000);
    LinearMaker<dtype, dimState> linear(1.0);
    FiniteVolumeSolver<dtype, dimState> solver(linear, "rusanov");

    // Initialize fields
    auto fields = solver.initialize(mesh, u_init);
    Field1D<dtype, dimState> Q1 = std::get<0>(fields);
    Field1D<dtype, dimState> Q2 = std::get<1>(fields);

    
    // Get solve step
    solveStep1D_ftype<dtype, dimState> solve_step = solver.get_solve_step(mesh);

    // Time loop
    float time = 0; float T_final = 0.75; float dt = 0.5*mesh.get_dx();

    while (time < T_final) {
        solve_step(Q1, Q2, dt);
        std::swap(Q1, Q2);
        time += dt;
    }
}
