#pragma once

#include <Eigen/Eigen>

typedef double Scalar;
typedef Eigen::Vector3<Scalar> Vector3s;
typedef Eigen::VectorX<Scalar> VectorXs;
typedef Eigen::MatrixX<Scalar> MatrixXs;
typedef Eigen::Triplet<Scalar> Triplet3s;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::Vector3i;

#define EPSILON 1e-15
#define block_vector(a) block<3,1>(3*(a), 0)

void initPlaneMesh(const Vector3s position,
    const Vector3s normal,
    MatrixXs& V, MatrixXi& SF,
    int resolution = 3, Scalar size = 1.0,
    const Vector3s tangent = Eigen::Vector3d(1.0, 0.0, 0.0));

void initSprings(const MatrixXs& V, MatrixXi& E, bool requestBending = false);

inline VectorXs _stack(const MatrixXs& m) {
    return VectorXs{ m.transpose().reshaped() };
}

inline MatrixXs _matri(VectorXs& v, int n_row, int n_col) {
    Eigen::Map<MatrixXs> m_T(v.data(), n_col, n_row);
    MatrixXs m(m_T.transpose());
    return m;
};