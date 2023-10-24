'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2023, Alexandre Didier, Jérôme Sieber, Rahel Rickenbach and Shao (Mike) Zhang, ETH Zurich,
% {adidier,jsieber, rrahel}@ethz.ch
%
% All rights reserved.
%
% This code is only made available for students taking the advanced MPC 
% class in the fall semester of 2023 (151-0371-00L) and is NOT to be 
% distributed.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from abc import ABC, abstractclassmethod

import numpy as np

from ampyc.utils import Polytope, qhull

# set random seed
np.random.seed(0)

class NoiseBase(ABC):
    """Base class for random noise/disturbance generators"""
    state_dependent = False

    def generate(self) -> np.ndarray:
        return self._generate()

    @abstractclassmethod
    def _generate(self) -> np.ndarray:
        """Noise generation method to be implemented by the inherited class"""
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        '''Calls the constructor again'''
        self.__init__(*args, **kwargs)


class ZeroNoise(NoiseBase):
    """Outputs zero noise, i.e., no disturbance"""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _generate(self) -> np.ndarray:
        return np.zeros((self.dim, 1))


class GaussianNoise(NoiseBase):
    """Computes Gaussian disturbance based on noise mean and covariance"""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        assert len(covariance.shape) == 2 and covariance.shape[0] == covariance.shape[1]
        assert len(mean) == covariance.shape[0]
        self.mean = mean.reshape(-1)
        self.cov = covariance

    def _generate(self) -> np.ndarray:
        return np.random.multivariate_normal(
            self.mean, self.cov, check_valid="raise"
        ).reshape(-1,1)


class TruncGaussianNoise(GaussianNoise):
    """Computes Gaussian disturbance based on noise mean and covariance in the set A_w * w <= b_w"""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, W: Polytope, max_iters: int = 1e4) -> None:
        super().__init__(mean, covariance)
        self.trunc_bounds = W
        self.max_iters = max_iters

    def _generate(self) -> np.ndarray:
        iters = 0
        w = super()._generate()
        while w not in self.trunc_bounds:
            w = super()._generate()
            iters += 1
            if iters > self.max_iters:
                raise Exception("exceeded max_iters of {0}, likely because of little overlap between the distribution and truncation polytope".format(self.max_iters))
        return w


class PolytopeVerticesNoise(NoiseBase):
    """Choses a random vertex of the vertix matrix as noise"""

    def __init__(self, W: Polytope) -> None:
        self.V = W.V

    def _generate(self) -> np.ndarray:
        idx = np.random.choice(self.V.shape[0])
        return self.V[idx, :].reshape(-1, 1)


class PolytopeNoise(NoiseBase):
    """Samples a random disturbance vector within a polytope, where vertices are weighted uniformly"""

    def __init__(self, W: Polytope) -> None:
        self.V = W.V

    def _generate(self) -> np.ndarray:
        """Based on implementation for randomPoint() in MPT"""
        L = np.random.uniform(size=(1, self.V.shape[0]))
        L /= np.sum(L)
        return (L @ self.V).reshape(-1, 1)


class StateDependentNoiseBase(NoiseBase):
    """Base class for state dependent random noise/disturbance generators"""
    state_dependent = True

    def generate(self, x: np.ndarray) -> np.ndarray:
        return self._generate(x)

    @abstractclassmethod
    def _generate(self, x: np.ndarray) -> np.ndarray:
        """Noise generation method to be implemented by the inherited class"""
        raise NotImplementedError
    

class StateDependentNoise(StateDependentNoiseBase):
    def __init__(self, G: np.ndarray) -> None:
        self.G = G

    def _generate(self, x: np.ndarray) -> np.ndarray:
        return (np.random.uniform() * self.G @ x).reshape(-1,1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def generate_N_times(noise_generator, N=500):
        x = []
        y = []
        for i in range(N):
            w = noise_generator.generate()
            x.append(w[0])
            y.append(w[1])
        return x, y

    """zero noise """
    zero_noise = ZeroNoise(dim=np.random.choice(10))
    for _ in range(100):
        assert np.all(zero_noise.generate() == 0.0)

    """gaussian noise """
    mean = np.array([0.5, 1.0]).reshape(-1, 1)
    covariance = np.diag([2.0, 0.5])
    gaussian_noise = GaussianNoise(mean, covariance)
    x, y = generate_N_times(gaussian_noise)

    plt.figure(1)
    plt.scatter(x, y)
    plt.grid()
    plt.title("gaussian noise")

    """truncated gaussian noise"""
    mean = np.array([0.5, 1.0]).reshape(-1, 1)
    covariance = np.diag([2.0, 0.5])
    A_w = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -0.0], [-0.0, -1.0]])
    b_w = np.array([2.0, 0.5, 0.0, 0.5])
    W = Polytope(A_w, b_w)
    trunc_gaussian_noise = TruncGaussianNoise(mean, covariance, W)
    trunc_gaussian_noise.generate()
    x, y = generate_N_times(trunc_gaussian_noise)

    plt.figure(2)
    plt.scatter(x, y)
    W.plot(ax=plt.gca(), alpha=0.5)
    plt.grid()
    plt.title("truncated gaussian noise")

    """generate a polytope"""
    x, y = generate_N_times(gaussian_noise, N=3)
    sampled_V = np.concatenate(
        [np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)], axis=1
    )
    W = qhull(sampled_V)

    """polytope vertex noise"""
    vertices_noise = PolytopeVerticesNoise(W)
    x, y = generate_N_times(vertices_noise)

    plt.figure(3)
    plt.scatter(x, y)
    W.plot(ax=plt.gca(), alpha=0.5)
    plt.grid()
    plt.title("polytope vertices noise")

    """polytope random noise"""
    polytope_noise = PolytopeNoise(W)
    x, y = generate_N_times(polytope_noise)

    plt.figure(4)
    plt.scatter(x, y)
    W.plot(ax=plt.gca(), alpha=0.5)
    plt.grid()
    plt.title("polytope uniform noise")


    """state dependent noise"""
    G = np.diag([0.0, 0.1])
    state_dependent_noise = StateDependentNoise(G)

    plt.show()
