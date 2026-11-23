"""NNTF uses the shared, unchanged tensor-factorization kernels. See _shared/training.py for its training flow."""
from baselines._shared.tensor_factorization import initialize, factor_step, objective, target_matrix
