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

    // Define mesh and initial condition
    Mesh1D mesh(1.0, 4000);
    fn_x_ftype<dimState> u_init = [](double x) {
        return 1.0;
    };

    // Define a linear model
    LinearMaker<dimState> model_maker(1.0);
    Model<dimState> model = model_maker.make_model();

    // Define a FV solver and a flux
    RusanovFlux<dimState> rusanov;
    FiniteVolumeSolver<dimState> solver(rusanov);

    // Initialize 1D field
    auto fields = solver.initialize(mesh, u_init);
    Field1D<dimState> Q = std::get<0>(fields);
    Field1D<dimState> Q_next = std::get<1>(fields);

    // Get solve step
    auto solve_step = solver.get_solve_step(mesh, model);

    // Dumb loop
    float time = 0.0; float dt = 0.5*mesh.get_dx(); float T_final = 0.75;
    while (time < T_final) {
        solve_step(Q, Q_next, dt);
        std::swap(Q, Q_next);
        time += dt;
    }
}
