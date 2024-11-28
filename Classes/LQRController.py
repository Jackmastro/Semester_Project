import numpy as np
from scipy.linalg import solve_discrete_are

class LQRController:
    """ Infinite horizon LQR controller """
    def __init__(self, A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray) -> None:
        # Check dimensions of A, B, Q, R
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if B.shape[0] != A.shape[0]:
            raise ValueError("Matrix B must have the same number of rows as A.")
        if Q.shape != A.shape:
            raise ValueError("Matrix Q must have the same dimensions as A.")
        if R.shape[0] != R.shape[1] or R.shape[0] != B.shape[1]:
            raise ValueError("Matrix R must be square and match the number of columns of B.")

        # Check if Q and R are (semi)positive definite
        if not self.is_positive_definite(Q):
            raise ValueError("Matrix Q must be positive semi-definite.")
        if not self.is_positive_definite(R):
            raise ValueError("Matrix R must be positive definite.")
        
        # Check controllability and observability
        if not self.is_controllable(A, B):
            raise ValueError("The system is not controllable with the given A and B matrices.")
        
        if not self.is_observable(A, C):
            raise ValueError("The system is not observable with the given A and C matrices.")

        # Solve Riccati equation
        self.P = solve_discrete_are(A, B, Q, R)
        self.K = -np.linalg.inv(R + B.T @ self.P @ B) @ B.T @ self.P @ A

    def get_control_input(self, current_state) -> np.ndarray:
        return self.K @ current_state

    @staticmethod
    def is_positive_definite(X:np.ndarray) -> bool:
        try:
            # Attempt Cholesky decomposition to check positive definiteness
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False
        
    @staticmethod
    def is_controllable(A, B):
        n = A.shape[0]
        controllability_matrix = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
        return np.linalg.matrix_rank(controllability_matrix) == n
    
    @staticmethod
    def is_observable(A, C):
        n = A.shape[0]
        observability_matrix = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
        return np.linalg.matrix_rank(observability_matrix) == n
