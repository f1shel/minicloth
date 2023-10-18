#include "Utils.h"

void initPlaneMesh(const Vector3s position,
    const Vector3s normal,
    MatrixXs& V, MatrixXi& SF,
    int resolution, Scalar size, const Vector3s tangent)
{
    int n_vertices = resolution * resolution;
    auto _y = normal.cross(tangent).normalized();
    auto _x = _y.cross(normal).normalized();
    auto step = size / (resolution - 1);
    V.resize(n_vertices, 3);
    for (int i = 0; i < n_vertices; i++) {
        int row_id = i / resolution;
        int col_id = i % resolution;
        auto v = position + step * col_id * _x + step * row_id * _y;
        V(i, 0) = v.x();
        V(i, 1) = v.y();
        V(i, 2) = v.z();
    }
    int n_squares = (resolution - 1) * (resolution - 1);
    int n_triangles = 2 * n_squares;
    SF.resize(n_triangles, 3);
    for (int i = 0; i < n_squares; i++) {
        int row_id = i / (resolution - 1);
        int col_id = i % (resolution - 1);
        int mode = (row_id + col_id) % 2;
        if (mode == 0) {
            // ___
            // |/|
            // ---
            SF(2 * i + 0, 0) = row_id * resolution + col_id;
            SF(2 * i + 0, 1) = row_id * resolution + col_id + 1;
            SF(2 * i + 0, 2) = (row_id + 1) * resolution + col_id;
            SF(2 * i + 1, 0) = row_id * resolution + col_id + 1;
            SF(2 * i + 1, 1) = (row_id + 1) * resolution + col_id + 1;
            SF(2 * i + 1, 2) = (row_id + 1) * resolution + col_id;
        }
        else {
            // ___
            // |\|
            // ---
            SF(2 * i + 0, 0) = row_id * resolution + col_id;
            SF(2 * i + 0, 1) = (row_id + 1) * resolution + col_id + 1;
            SF(2 * i + 0, 2) = (row_id + 1) * resolution + col_id;
            SF(2 * i + 1, 0) = row_id * resolution + col_id;
            SF(2 * i + 1, 1) = row_id * resolution + col_id + 1;
            SF(2 * i + 1, 2) = (row_id + 1) * resolution + col_id + 1;
        }
    }

}

void initSprings(const MatrixXs& V, MatrixXi& E, bool requestBending) {
    int n_vertices = static_cast<int>(V.rows());
    int resolution = static_cast<int>(sqrtf(n_vertices));

    int n_structural = 2 * (resolution - 1) * resolution;
    int n_shearing = (resolution - 1) * (resolution - 1);
    int n_bending = requestBending ? 2 * (resolution - 2) * resolution: 0;

    E.resize(n_structural + n_shearing + n_bending, 2);

    // structural springs
    for (int i = 0; i < n_structural / 2; i++) {
        int row_id = i / (resolution - 1);
        int col_id = i % (resolution - 1);
        // horizontal
        E(2 * i + 0, 0) = row_id * resolution + col_id;
        E(2 * i + 0, 1) = row_id * resolution + col_id + 1;
        // vertical
        E(2 * i + 1, 0) = col_id * resolution + row_id;
        E(2 * i + 1, 1) = (col_id + 1) * resolution + row_id;
    }

    // shearing springs
    for (int i = 0; i < n_shearing; i++) {
        int row_id = i / (resolution - 1);
        int col_id = i % (resolution - 1);
        int mode = (row_id + col_id) % 2;
        if (mode == 0) {
            // ___
            // |/|
            // ---
            E(n_structural + i, 0) = row_id * resolution + col_id + 1;
            E(n_structural + i, 1) = (row_id + 1) * resolution + col_id;
        }
        else {
            // ___
            // |\|
            // ---
            E(n_structural + i, 0) = row_id * resolution + col_id;
            E(n_structural + i, 1) = (row_id + 1) * resolution + col_id + 1;
        }
    }

    // bending springs
    for (int i = 0; i < n_bending / 2; i++) {
        int row_id = i / (resolution - 2);
        int col_id = i % (resolution - 2);
        // horizontal
        E(n_structural + n_shearing + 2 * i + 0, 0) = row_id * resolution + col_id;
        E(n_structural + n_shearing + 2 * i + 0, 1) = row_id * resolution + col_id + 2;
        // vertical
        E(n_structural + n_shearing + 2 * i + 1, 0) = col_id * resolution + row_id;
        E(n_structural + n_shearing + 2 * i + 1, 1) = (col_id + 2) * resolution + row_id;
    }

}