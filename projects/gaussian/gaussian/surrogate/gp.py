import math
import numpy as np

from typing import Tuple, List, Optional, Any, Required, Callable
from numpy.typing import NDArray
from numpy import float64


def squared_exponential(
        x1: NDArray[float64], 
        x2: NDArray[float64],
        l: float64,
        noise_var: float,
        ) -> float64:
    if x1.shape != x2.shape:
        raise ValueError(f"Shape of x1 {x1.shape}, does not match shape of x2: {x2.shape}!")
    return np.exp(
        - np.sum(np.square(np.abs(x1 - x2))) / (2 * l * l)
    )

def cov_matrix(
        mat1: NDArray[float64],
        mat2: NDArray[float64],
        l: float64,
        noise_var: float,

        kernel: Callable
    ) -> NDArray[float64]:
    n: int = mat1.shape[0]
    m: int = mat2.shape[0]

    K: NDArray[float64] = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i][j] = kernel(
                x1=np.transpose(mat1[i], axes=None),
                x2=np.transpose(mat2[j], axes=None),
                l=l,
                noise_var=noise_var
            )
    return K

def inv_kernel(
        mat1: NDArray[float64],
        l: float,
        noise_var: float64,
        
        kernel: Callable
    ) -> NDArray[float64]:

    K_mat1_mat1: NDArray[float64] = cov_matrix(
        mat1=mat1,
        mat2=mat1,
        l=l,
        noise_var=noise_var,

        kernel=kernel
    )

    K_dim0: int = K_mat1_mat1.shape[0]
    noise_identity: NDArray[float64] = np.identity(n=K_dim0)

    return np.linalg.inv(K_mat1_mat1 + noise_identity * noise_var)

def f_mean(
        mat1: NDArray[float64], # dataset
        mat2: NDArray[float64], # training set
        inv: NDArray[float64],  # inverse

        y: NDArray[float64], # observations
        l: float,
        noise_var: float,

        kernel: Callable,
    ):

    K_mat2_mat1: NDArray[float64] = cov_matrix(
        mat1=mat2,
        mat2=mat1,
        l=l,
        noise_var=noise_var,

        kernel=kernel
    )

    return np.matmul(np.matmul(K_mat2_mat1, inv), y)

def f_cov(
        mat1: NDArray[float64],
        mat2: NDArray[float64],
        inv: NDArray[float64],

        l: float,
        noise_var: float,

        kernel: Callable,
    ):

    K_mat2_mat2: NDArray[float64] = cov_matrix(
        mat1=mat2,
        mat2=mat2,

        l=l,
        noise_var=noise_var,

        kernel=kernel
    )
    K_mat2_mat1: NDArray[float64] = cov_matrix(
        mat1=mat2,
        mat2=mat1,

        l=l,
        noise_var=noise_var,

        kernel=kernel
    )
    K_mat1_mat2: NDArray[float64] = K_mat2_mat1.T # mat1_mat2 is symmetrical, thus K_mat1_mat2-1 = K_mat2_mat1

    return K_mat2_mat2 - np.matmul(np.matmul(K_mat2_mat1, inv), K_mat1_mat2)


def lmm(
        mat1: NDArray[float64],
        inv: NDArray[float64],
        y: NDArray[float64],
        noise_var: float,

        log_func: Callable = math.log
        ):
    n: int = mat1.shape[0]

    transpose: NDArray[float64] = (-1/2) * np.matmul(np.matmul(y.T, inv), y)

    noise_identity: NDArray[float64] = np.identity(n=n)
    log: NDArray[float64] = (-1/2) * log_func(np.abs(mat1 + noise_identity * noise_var))

    norm: NDArray[float64] = -(n/2) * log_func(2 * math.pi)

    return transpose + log + norm

def gaussian_surrogate(
        mat1: NDArray[float64],
        mat2: NDArray[float64],
        y: NDArray[float64],
        kernel: Callable,
        l: float,
        noise_var: float
        ) -> Tuple[NDArray[float64], NDArray[float64]]:
    
    inv: NDArray[float64] = inv_kernel(
        mat1=mat1,
        l=l,
        noise_var=noise_var,
        kernel=kernel
        )
    f_m: NDArray[float64] = f_mean(
        mat1=mat1,
        mat2=mat2,
        inv=inv,

        y=y,
        l=l,
        noise_var=noise_var,

        kernel=kernel,
    )
    f_c: NDArray[float64] = f_cov(
        mat1=mat1,
        mat2=mat2,
        inv=inv,
        
        l=l,
        noise_var=noise_var,

        kernel=kernel
    )

    f_lmm: NDArray[float64] = lmm(
        mat1=mat1,
        inv=inv,
        y=y,

        noise_var=noise_var,
    )

    return (f_m, f_c, f_lmm)