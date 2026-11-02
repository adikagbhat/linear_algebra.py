"""
Numerical Computing & Linear Algebra Utilities
Deterministic solvers for system of linear equations and decompositions.
"""
import numpy as np


def lu_decomposition(matrix: np.ndarray):
    """
    Performs Doolittle LU decomposition.
    A = L * U
    """
    n = matrix.shape[0]
    L = np.eye(n, dtype=float)
    U = matrix.astype(float).copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    return L, U


def solve_lu(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward and backward substitution to solve Ax = b."""
    n = len(b)
    # Forward substitution Ly = b
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Backward substitution Ux = y
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x


def verify_residual(
    A: np.ndarray, x: np.ndarray, b: np.ndarray, tol: float = 1e-6
) -> bool:
    """Computes L2 norm residual error."""
    residual = np.linalg.norm(np.dot(A, x) - b, ord=2)
    return bool(residual < tol)


if __name__ == "__main__":
    test_A = np.array([[4.0, 3.0], [6.0, 3.0]])
    test_b = np.array([10.0, 12.0])
    L, U = lu_decomposition(test_A)
    sol = solve_lu(L, U, test_b)
    print("Computed solution x:", sol)
    print("Residual verified:", verify_residual(test_A, sol, test_b))
