#include <iostream>
#include <tuple>
#include <eigen3/Eigen/Dense>

#include "mesh/mesh1D.hpp"
#include "utils/linear.hpp"
#include "utils/typdefs.hpp"
#include "utils/timer.hpp"
#include "models.hpp"
#include "solver.hpp"
using namespace FVTYPES;

int main() {
    // Define dtype and dimState
    using dtype = double;
    const unsigned int dimState = 1;
    using fieldType = Field2D<dtype, dimState>;

    // Initial condition
    fnXY_ftype<dtype, dimState> u_init = [](xyPoint p) {
        //return std::array<dtype, dimState>({1.0, 1.0});
        return 1.0;
    };

    // Define model and solver
    Mesh2D mesh(1.0, 1.0, 512, 512);
    //EigType::Matrix<dtype, dimState> A_flux( {{1.0, 0.0},{0.0, 1.0}} );
    dtype A_flux = 1.0;
    LinearMaker<dtype, dimState> linear( A_flux );
    FiniteVolumeSolver<dtype, dimState> solver(linear, "rusanov");

    // Initialize field
    fieldType Q = solver.initialize(mesh, u_init);
    fieldType Q_next = solver.initialize(mesh, u_init);

    // Get solve step
    solveStep2D_ftype<dtype, dimState> solve_step = solver.get_solve_step(mesh);

    // Time loop
    float T_final = 0.75; float dt = 0.5*mesh.get_dx(); float time = 0.0;

    Timer timer;
    fieldType Q_final = Integrator::Euler< fieldType >(solve_step, Q, T_final, dt);
    timer.stop();

    std::cout << ">> Time : " << timer.getElapsed() << std::endl;

    std::string IMAGE_PATH = "/home/antoine/Internship/fvpackage/results";
    Q_final.save_to_csv(mesh, IMAGE_PATH + '/' + "2D-vector.csv");
}
