#include <igl/opengl/glfw/Viewer.h>
#include <igl/stb/write_image.h>
#include "Utils.h"
#include <iostream>
#include <cstdlib>

int resolution   = 64;
Scalar mass      = 1e2 / (resolution * resolution);
Scalar dt        = 1.0 / 30;
Scalar stiffness = 3e3;
Scalar damping   = 0.995;

int n_springs;        // number of springs
int n_vertices;       // number of vertices
MatrixXs V;           // vertices position
MatrixXi SF;          // surface ids
MatrixXi E;           // edge ids
VectorXs rest_length; // rest length of springs
VectorXs lVel;        // linear velocity

// initialization and precompute is done in first frame
bool first_frame = true;

void captureScreen(igl::opengl::glfw::Viewer& viewer, std::string savePath) {
    // Allocate temporary buffers for 1280x800 image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(1280, 800);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(1280, 800);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(1280, 800);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(1280, 800);
    // Draw the scene in the buffers
    viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);
    // Save it to a PNG
    igl::stb::write_image(savePath, R, G, B, A);
}

void explicitIntegration() { // crash
    lVel *= damping;
    VectorXs force(3 * n_vertices);
    force.setZero();
    for (int i = 0; i < n_springs; i++) {
        int i0 = E(i, 0), i1 = E(i, 1);
        auto x01 = V.row(i0) - V.row(i1);
        Vector3s elastic_force = stiffness * (1 - rest_length(i) / x01.norm()) * x01;
        force.block_vector(i0) += -elastic_force;
        force.block_vector(i1) += elastic_force;
    }
    for (int i = 0; i < n_vertices; i++) {
        if (i == 0 || i == resolution - 1) continue;
        lVel.block_vector(i) += force.block_vector(i) / mass * dt;
        lVel(3 * i + 1) += -9.8 * dt;
        V.row(i) += lVel.block_vector(i) * dt;
    }
}

void diagonalDescentWithoutLineSearch() { // crash
    VectorXs gradient(3 * n_vertices);
    VectorXs current_x = _stack(V);
    VectorXs inertia_y = current_x + damping * dt * lVel;
    VectorXs next_x = inertia_y;
    Scalar inv_dt2 = 1 / (dt * dt);
    for (int k = 0; k < 32; k++) {
        auto inertia = next_x - inertia_y;
        // calculate gradient
        for (int i = 0; i < n_vertices; i++) {
            gradient.block_vector(i) = -inv_dt2 * mass * inertia.block_vector(i);
            gradient(3 * i + 1) += mass * -9.8;
        }
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            auto x01 = next_x.block_vector(i0) - next_x.block_vector(i1);
            auto elastic_force = stiffness * (1 - rest_length(i) / x01.norm()) * x01;
            gradient.block_vector(i0) += -elastic_force;
            gradient.block_vector(i1) += elastic_force;
        }
        // calculate hessian and descent direction
        for (int i = 0; i < n_vertices; i++) {
            if (i == 0 || i == resolution - 1) continue;
            auto quasi_hessian = inv_dt2 * mass + 4 * stiffness;
            auto delta = 1.0 / quasi_hessian * gradient.block_vector(i);
            next_x.block_vector(i) += delta;
        }
    }
    for (int i = 0; i < n_vertices; i++) {
        if (i == 0 || i == resolution - 1) continue;
        V.row(i) = next_x.block_vector(i);
        lVel.block_vector(i) = (next_x.block_vector(i) - current_x.block_vector(i)) / dt;
    }
}

void fastMassSpringQuasiWithLineSearch() {
    static Eigen::SparseMatrix<Scalar> L, J, M, H;
    static Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>> system_matrix;

    VectorXs current_x = _stack(V);
    VectorXs inertia_y = current_x + damping * dt * _stack(lVel);
    inertia_y.block_vector(0) = current_x.block_vector(0);
    inertia_y.block_vector(resolution - 1) = current_x.block_vector(resolution - 1);
    VectorXs next_x = inertia_y;
    VectorXs g = Vector3s(0, -9.8, 0).replicate(n_vertices, 1);

    if (first_frame) {
        first_frame = false;
        // -- M
        std::vector<Triplet3s> MTriplets;
        M.resize(3 * n_vertices, 3 * n_vertices);
        MTriplets.clear();
        for (int i = 0; i < n_vertices; i++)
            for (int j = 0; j < 3; j++)
                MTriplets.emplace_back(Triplet3s(3 * i + j, 3 * i + j, mass));
        M.setFromTriplets(MTriplets.begin(), MTriplets.end());
        // -- L
        std::vector<Triplet3s> LTriplets;
        L.resize(3 * n_vertices, 3 * n_vertices);
        LTriplets.clear();
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            for (int j = 0; j < 3; j++) {
                LTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i0 + j, 1 * stiffness));
                LTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i1 + j, 1 * stiffness));
                LTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i1 + j, -1 * stiffness));
                LTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i0 + j, -1 * stiffness));
            }
        }
        L.setFromTriplets(LTriplets.begin(), LTriplets.end());
        // -- J
        std::vector<Triplet3s> JTriplets;
        J.resize(3 * n_vertices, 3 * n_springs); // extra 2 fixed points constraints
        JTriplets.clear();
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            for (int j = 0; j < 3; j++) {
                JTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i + j, 1 * stiffness));
                JTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i + j, -1 * stiffness));
            }
        }
        J.setFromTriplets(JTriplets.begin(), JTriplets.end());
        // -- Hessian
        H = M + dt * dt * L;
        system_matrix.compute(H);
    }

    for (int k = 0; k < 64; k++) {
        // local step
        VectorXs p(3 * n_springs);
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            Vector3s x0 = next_x.block_vector(i0), x1 = next_x.block_vector(i1);
            Vector3s p12 = (x0 - x1).normalized();
            p.block_vector(i) = p12 * rest_length(i);
        }
        // global step
        VectorXs gradient = M * (next_x - inertia_y) + dt * dt * (L * next_x - J * p - M * g);
        gradient.block_vector(0) *= 0.0;
        gradient.block_vector(resolution - 1) *= 0.0;
        VectorXs descent_dir = -system_matrix.solve(gradient);
        descent_dir.block_vector(0) *= 0.0;
        descent_dir.block_vector(resolution - 1) *= 0.0;

        // line search
        Scalar ls_step_size = 1.0;
        Scalar ls_alpha = 0.25;
        Scalar ls_beta = 0.1;
        auto evaluateObjectiveFunction = [&](const VectorXs& x)
        {
            Scalar potential_term = 0.0;

            for (int i = 0; i < n_springs; i++) {
                int i0 = E(i, 0), i1 = E(i, 1);
                Vector3s x_01 = x.block_vector(i0) - x.block_vector(i1);
                Scalar delta_l = x_01.norm() - rest_length(i);
                Scalar e = 0.5 * stiffness * delta_l * delta_l;
                potential_term += e;
            }

            // external force
            potential_term -= x.transpose() * M * g;

            Scalar inertia_term = 0.5 * (x - inertia_y).transpose() * M * (x - inertia_y);

            return inertia_term + potential_term * dt * dt;
        };
        auto lineSearch = [&](const VectorXs& x, const VectorXs& gradient_dir, const VectorXs& descent_dir) {
            VectorXs x_plus_tdx(3 * n_vertices);
            Scalar t = 1.0 / ls_beta, lhs, rhs;

            Scalar currentObjectiveValue = evaluateObjectiveFunction(x);

            do
            {
                t *= ls_beta;
                x_plus_tdx = x + t * descent_dir;

                lhs = evaluateObjectiveFunction(x_plus_tdx);
                rhs = currentObjectiveValue + ls_alpha * t * (gradient_dir.transpose() * descent_dir)(0);

            } while (lhs >= rhs && t > EPSILON);

            if (t < EPSILON) t = 0.0;

            return t;
        };
        Scalar step_size = lineSearch(next_x, gradient, descent_dir);

        next_x = next_x + step_size * descent_dir;
    }
    V = _matri(next_x, n_vertices, 3);
    lVel = (next_x - current_x) / dt;
}

void fastMassSpring() {
    static Eigen::SparseMatrix<Scalar> L, J, M, H;
    static Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>> system_matrix;
    static Vector3s p0, p1; // 2 fixed position

    VectorXs current_x = _stack(V);
    VectorXs inertia_y = current_x + damping * dt * _stack(lVel);
    VectorXs next_x = inertia_y;
    VectorXs g = Vector3s(0, -9.8, 0).replicate(n_vertices, 1);

    if (first_frame) {
        first_frame = false;
        // -- M
        std::vector<Triplet3s> MTriplets;
        M.resize(3 * n_vertices, 3 * n_vertices);
        MTriplets.clear();
        for (int i = 0; i < n_vertices; i++)
            for (int j = 0; j < 3; j++)
                MTriplets.emplace_back(Triplet3s(3 * i + j, 3 * i + j, mass));
        M.setFromTriplets(MTriplets.begin(), MTriplets.end());
        // -- L
        std::vector<Triplet3s> LTriplets;
        L.resize(3 * n_vertices, 3 * n_vertices);
        LTriplets.clear();
        for (int j = 0; j < 3; j++) {
            // extra stiffer springs for fix points
            int i0 = 0, i1 = resolution - 1;
            LTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i0 + j, 1 * 100 * stiffness));
            LTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i1 + j, 1 * 100 * stiffness));
        }
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            for (int j = 0; j < 3; j++) {
                LTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i0 + j, 1 * stiffness));
                LTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i1 + j, 1 * stiffness));
                LTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i1 + j, -1 * stiffness));
                LTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i0 + j, -1 * stiffness));
            }
        }
        L.setFromTriplets(LTriplets.begin(), LTriplets.end());
        // -- J
        std::vector<Triplet3s> JTriplets;
        J.resize(3 * n_vertices, 3 * (n_springs + 2)); // extra 2 fixed points constraints
        JTriplets.clear();
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0);
            int i1 = E(i, 1);
            for (int j = 0; j < 3; j++) {
                JTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * i + j, 1 * stiffness));
                JTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * i + j, -1 * stiffness));
            }
        }
        for (int j = 0; j < 3; j++) {
            int i0 = 0, i1 = resolution - 1;
            JTriplets.emplace_back(Triplet3s(3 * i0 + j, 3 * (n_springs + 0) + j, 1 * 100 * stiffness));
            JTriplets.emplace_back(Triplet3s(3 * i1 + j, 3 * (n_springs + 1) + j, 1 * 100 * stiffness));
        }
        J.setFromTriplets(JTriplets.begin(), JTriplets.end());
        p0 = V.row(0);
        p1 = V.row(resolution - 1);
        // -- Hessian
        H = M + dt * dt * L;
        system_matrix.compute(H);
    }

    for (int k = 0; k < 10; k++) {
        // local step
        VectorXs p(3 * (n_springs + 2));
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            Vector3s x0 = next_x.block_vector(i0), x1 = next_x.block_vector(i1);
            Vector3s p12 = (x0 - x1).normalized();
            p.block_vector(i) = p12 * rest_length(i);
        }
        p.block_vector(n_springs) = p0;
        p.block_vector(n_springs + 1) = p1;
        // global step
        VectorXs right_hand = M * inertia_y + dt * dt * J * p + dt * dt * M * g;
        next_x = system_matrix.solve(right_hand);
    }
    V = _matri(next_x, n_vertices, 3);
    lVel = (next_x - current_x) / dt;
}

void newtonDescent() {
    static Eigen::SparseMatrix<Scalar> M;
    static VectorXs gravity;

    if (first_frame) {
        first_frame = false;
        // -- M
        std::vector<Triplet3s> MTriplets;
        M.resize(3 * n_vertices, 3 * n_vertices);
        MTriplets.clear();
        for (int i = 0; i < n_vertices; i++)
            for (int j = 0; j < 3; j++)
                MTriplets.emplace_back(Triplet3s(3 * i + j, 3 * i + j, mass));
        M.setFromTriplets(MTriplets.begin(), MTriplets.end());
        // -- gravity force
        gravity = Vector3s(0, -9.8, 0).replicate(n_vertices, 1);
        gravity = M * gravity;
    }

    VectorXs current_x = _stack(V);
    // calculate inertia y
    VectorXs inertia_y = current_x + dt * lVel;
    inertia_y.block_vector(0) = current_x.block_vector(0);
    inertia_y.block_vector(resolution - 1) = current_x.block_vector(resolution - 1);
    // take a initial guess
    VectorXs next_x = inertia_y;

    VectorXs gradient(3 * n_vertices);
    for (int k = 0; k < 32; k++) {
        // evaluate gradient
        gradient = -gravity;
        for (int i = 0; i < n_springs; i++) {
            // minus elastic spring force
            int i0 = E(i, 0), i1 = E(i, 1);
            auto x01 = next_x.block_vector(i0) - next_x.block_vector(i1);
            auto elastic_force = stiffness * (1 - rest_length(i) / x01.norm()) * x01;
            gradient.block_vector(i0) -= -elastic_force;
            gradient.block_vector(i1) -= elastic_force;
        }

        gradient = mass * (next_x - inertia_y) + dt * dt * gradient;

        gradient.block_vector(0) *= 0.0;
        gradient.block_vector(resolution - 1) *= 0.0;

        if (gradient.squaredNorm() < EPSILON)
            break;

        // evaluate hessian matrix
        Eigen::SparseMatrix<Scalar> H(3 * n_vertices, 3 * n_vertices);
        std::vector<Triplet3s> HTriplets;
        HTriplets.clear();
        // -- spring hessian
        for (int i = 0; i < n_springs; i++) {
            int i0 = E(i, 0), i1 = E(i, 1);
            Vector3s x_01 = next_x.block_vector(i0) - next_x.block_vector(i1);
            Scalar l_01 = x_01.norm();
            Scalar r = rest_length(i);

            auto I = Eigen::Matrix<Scalar, 3, 3>::Identity();
            auto tmp = x_01 * x_01.transpose() / (l_01 * l_01);
            Eigen::Matrix<Scalar, 3, 3> small_hessian = stiffness * (I - r / l_01 * (I - (x_01 * x_01.transpose()) / (l_01 * l_01)));
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    Scalar val = small_hessian(row, col);
                    HTriplets.push_back(Triplet3s(3 * i0 + row, 3 * i0 + col, val));
                    HTriplets.push_back(Triplet3s(3 * i0 + row, 3 * i1 + col, -val));
                    HTriplets.push_back(Triplet3s(3 * i1 + row, 3 * i0 + col, -val));
                    HTriplets.push_back(Triplet3s(3 * i1 + row, 3 * i1 + col, val));
                }
            }
        }
        H.setFromTriplets(HTriplets.begin(), HTriplets.end());

        H = M + dt * dt * H;

        // solve descent dir
        auto factorizeDirectSolverLDLT = [](
            const Eigen::SparseMatrix<Scalar>& A,
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Upper>& ldltSolver)
        {
            Eigen::SparseMatrix<Scalar> A_prime = A;
            ldltSolver.analyzePattern(A_prime);
            ldltSolver.factorize(A_prime);
            Scalar Regularization = 0.00001;
            bool success = true;
            Eigen::SparseMatrix<Scalar> I;
            I.resize(3 * n_vertices, 3 * n_vertices);
            I.setIdentity();
            while (ldltSolver.info() != Eigen::Success)
            {
                Regularization *= 10;
                A_prime = A_prime + Regularization * I;
                ldltSolver.factorize(A_prime);
                success = false;
            }
            if (!success)
                std::cout << "Warning: " << "Newton Descent" << " adding " << Regularization << " identites.(ldlt solver)" << std::endl;
        };
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Upper> ldltSolver;
        factorizeDirectSolverLDLT(H, ldltSolver);
        VectorXs descent_dir = -ldltSolver.solve(gradient);

        descent_dir.block_vector(0) *= 0.0;
        descent_dir.block_vector(resolution - 1) *= 0.0;

        // line search
        Scalar ls_step_size = 1.0;
        Scalar ls_alpha = 0.25;
        Scalar ls_beta = 0.1;
        auto evaluateObjectiveFunction = [&](const VectorXs& x)
        {
            Scalar potential_term = 0.0;

            for (int i = 0; i < n_springs; i++) {
                int i0 = E(i, 0), i1 = E(i, 1);
                Vector3s x_01 = x.block_vector(i0) - x.block_vector(i1);
                Scalar delta_l = x_01.norm() - rest_length(i);
                Scalar e = 0.5 * stiffness * delta_l * delta_l;
                potential_term += e;
            }

            // external force
            potential_term -= x.transpose() * gravity;

            Scalar inertia_term = 0.5 * (x - inertia_y).transpose() * M * (x - inertia_y);

            return inertia_term + potential_term * dt * dt;
        };
        auto lineSearch = [&](const VectorXs& x, const VectorXs& gradient_dir, const VectorXs& descent_dir) {
            VectorXs x_plus_tdx(3 * n_vertices);
            Scalar t = 1.0 / ls_beta, lhs, rhs;

            Scalar currentObjectiveValue = evaluateObjectiveFunction(x);

            do
            {
                t *= ls_beta;
                x_plus_tdx = x + t * descent_dir;

                lhs = evaluateObjectiveFunction(x_plus_tdx);
                rhs = currentObjectiveValue + ls_alpha * t * (gradient_dir.transpose() * descent_dir)(0);

            } while (lhs >= rhs && t > EPSILON);

            if (t < EPSILON) t = 0.0;

            return t;
        };
        Scalar step_size = lineSearch(next_x, gradient, descent_dir);

        // update x
        next_x = next_x + descent_dir * step_size;
    }
    V = _matri(next_x, n_vertices, 3);
    lVel = (next_x - current_x) / dt;
}

////////////////////////////////////////////////////////////////////////
// !!!!!!!!!!! SELECT METHOD HERE !!!!!!!!!!!
void (*proceedOptimization)(void) = fastMassSpringQuasiWithLineSearch;
////////////////////////////////////////////////////////////////////////

bool drawFunc(igl::opengl::glfw::Viewer& viewer) {
    proceedOptimization();
    // update mesh data
    MatrixXs V_vis = V;
    viewer.data().set_mesh(V_vis, SF);
    viewer.data().compute_normals();
    viewer.data().set_face_based(true);
    viewer.core().lighting_factor = 0.0;

    static int frame_cnt = 0;
    char fp[100];
    sprintf(fp, "./fast/frame_%04d.png", frame_cnt++);
    captureScreen(viewer, fp);

    return false;
}

int main(int argc, char *argv[])
{
    Vector3s position(0.0, 0.0, 0.0);
    Vector3s normal(0.0, 1.0, 0.0);

    initPlaneMesh(position, normal, V, SF, resolution);
    initSprings(V, E, true);

    // init rest length
    n_springs = static_cast<int>(E.rows());
    rest_length.resize(n_springs, 1);
    for (int i = 0; i < n_springs; i++) {
        int i0 = E(i, 0);
        int i1 = E(i, 1);
        rest_length(i) = (V.row(i0) - V.row(i1)).norm();
    }

    // init velocity
    n_vertices = static_cast<int>(V.rows());
    lVel.resize(3 * n_vertices);
    lVel.setZero();

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.core().background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.core().is_animating = true;
    MatrixXs VV(2, 3);
    VV.row(0) = Vector3s(0, 0, 0);
    VV.row(1) = Vector3s(1, -1, 0);
    viewer.core().align_camera_center(VV);
    viewer.core().camera_eye.z() *= -1;
    viewer.core().camera_eye.x() -= 4.3;
    viewer.core().camera_eye.y() += 2;
    viewer.core().camera_eye.y() -= 0.5;
    viewer.core().camera_center.y() -= 0.5;
    viewer.core().camera_zoom = 1.5;
    viewer.callback_pre_draw = &drawFunc;
    viewer.launch();
}
