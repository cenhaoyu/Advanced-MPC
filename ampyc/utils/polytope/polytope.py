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

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

import polytope as pc
from polytope.polytope import projection, reduce, extreme, is_fulldim, _get_patch
from polytope.quickhull import quickhull


### AMPC Polytope Class ###

class Polytope(pc.Polytope):
    '''Improved Polytope class with additional functionality compared to polytope.Polytope'''

    def __init__(self, A=np.array([]), b=np.array([]), vertices=None, **kwargs):
        super().__init__(A=A, b=b, vertices=vertices, normalize=False, **kwargs)
        
        # always compute V representation (comment out for better performance)
        if self.vertices is None and self.dim > 0:
           self.vertices = extreme(self)

        # alias for vertices
        self.V = self.vertices

        # get bounding box
        if self.dim > 0:
            self.bounding_box

        if self.bbox is not None:
            box = np.array(self.bbox)
            self.xlim = [box[0,0].item()*1.1, box[1,0].item()*1.1]
            if self.dim > 1:
                self.ylim = [box[0,1].item()*1.1, box[1,1].item()*1.1]

    __array_ufunc__ = None  # disable numpy ufuncs

    def __add__(self, other):
        if isinstance(other, Polytope):
            if self.vertices is None:
                self.vertices = extreme(self)
            if other.vertices is None:
                other.vertices = extreme(other)
            return _minkowski_sum(self, other)
        else:
            return Polytope(A=self.A, b=self.b.reshape(-1,1) + self.A@other.reshape(-1,1))
    
    def __radd__(self, other):
        return Polytope(A=self.A, b=self.b.reshape(-1,1) + self.A@other.reshape(-1,1))
    
    def __and__(self, other):
        return self.intersect(other)
    
    def __sub__(self, other):
        if isinstance(other, Polytope):
            if self.vertices is None:
                self.vertices = extreme(self)
            if other.vertices is None:
                other.vertices = extreme(other)
            return _pontryagin_difference(self, other)
        else:
            return Polytope(A=self.A, b=self.b.reshape(-1,1) - self.A@other.reshape(-1,1))
        
    def __rsub__(self, other):
        return Polytope(A=-self.A, b=self.b.reshape(-1,1) - self.A@other.reshape(-1,1))
    
    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise NotImplementedError('Product of two polytopes is not well defined')
        else:
            return _scale_polytope(other, self)
        
    def __rmul__(self, other):
        if not isinstance(other, (float, int)):
            raise NotImplementedError('Product of two polytopes is not well defined')
        else:
            return _scale_polytope(other, self)
        
    def __matmul__(self, other):
        raise NotImplementedError('Right matrix multiplication is not defined for Polytopes')
        
    def __rmatmul__(self, other):
        if not isinstance(other, np.ndarray):
            raise NotImplementedError('Product of two polytopes is not well defined')
        else:
            if self.vertices is None:
                self.vertices = extreme(self)
            return _matrix_propagate_polytope(other, self)
    
    def grid(self, N=10):
        bbox = np.hstack(self.bbox)
        XX = np.linspace(bbox[0,0],bbox[0,1], int(np.floor(np.sqrt(N))))
        YY = np.linspace(bbox[1,0],bbox[1,1], int(np.floor(np.sqrt(N))))
        return np.stack(np.meshgrid(XX, YY),axis=2)
        
    def intersect(self, other):
        P = super().intersect(other)
        return Polytope(A=P.A, b=P.b, vertices=P.vertices)
    
    def plot(self, ax=None, alpha=0.25, color=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if not is_fulldim(self):
            raise RuntimeError("Cannot plot empty polytope")

        poly = _get_patch(
            self, facecolor=color, alpha=alpha, **kwargs)
        poly.set_zorder(2) # we need this because _get_patch sets zorder to 0
        ax.add_patch(poly)
        return ax
    
    def project(self, dim, solver=None, abs_tol=1e-7, verbose=0):
        """Return Polytope projection on selected subspace.

        For usage details see function: L{_projection}.
        """
        return _projection(self, dim, solver, abs_tol, verbose)
    
    def support(self, eta):
        return _support(self, eta)
    
    def Vrep(self):
        if self.vertices is None:
            self.vertices = extreme(self)
            self.V = self.vertices
            return self.vertices
        else:
            return self.vertices
    
def qhull(vertices: np.array, abs_tol: float = 1e-7, verbose: bool = False) -> Polytope:
    """Use quickhull to compute a convex hull.

    @param vertices: A N x d array containing N vertices in dimension d

    @return: L{Polytope} describing the convex hull
    """
    dim = vertices.shape[1]
    rays = vertices - vertices[0, :]
    _,S,_ = np.linalg.svd(rays)

    if np.any(S < abs_tol):
        if verbose:
            print("[Warning] degenerate polytope detected! Cannot compute polytope in H-Rep; returning Polytope only in V-Rep!")
        # remove redundant vertices; only works for 2D case!
        if dim == 2:
            redundant_idx = np.all(np.abs(np.cross(rays[1:, None, :], rays[1:])) < abs_tol, axis=0)
            if np.all(redundant_idx):
                # all vertices are redundant; keep the vertex with largest distance to the first
                keep_idx = np.argmax(np.linalg.norm(rays, axis=1))
                vert = np.vstack([vertices[0], vertices[keep_idx]])
            else:
                # remove redundant vertices
                vert = np.vstack([vertices[0], vertices[1:][~redundant_idx]])
        else:
            # for higher dimensions, return original vertices
            vert = vertices
        
        return Polytope(vertices=vert)
    else:
        A, b, vert = quickhull(vertices, abs_tol=abs_tol)
        if A.size == 0:
            if verbose:
                print("[Warning] Could not find convex hull; returning empty polytope!")
            return Polytope()
        
        return Polytope(A, b, minrep=True, vertices=vert)

def _reduce(P: Polytope) -> Polytope:
    """This is just a wrapper around the polytope.reduce function."""
    pc_P = reduce(P)
    return Polytope(A=pc_P.A, b=pc_P.b, vertices=pc_P.vertices)
    
def _support(P: Polytope, eta: np.array) -> float:
    '''
    The support function of the polytope P, evaluated at (or in the direction)
    eta in R^n

    Based on https://github.com/heirung/pytope/blob/master/pytope/polytope.py#L457
    '''
    n = P.A.shape[1]
    x = cp.Variable((n,1))
    constraints = [P.A @ x <= P.b.reshape(-1,1)]
    objective = cp.Maximize(eta.reshape(1,-1) @ x)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != 'optimal':
        raise Exception('Unable to compute support for the given polytope and direction eta!')
    return objective.value

def _minkowski_sum(P: Polytope, Q: Polytope) -> Polytope:
    '''
    Minkowski sum of two convex polytopes P and Q:
    P + Q = {p + q in R^n : p in P, q in Q}.
    In vertex representation, this is the convex hull of the pairwise sum of all
    combinations of points in P and Q.

    Based on https://github.com/heirung/pytope/blob/master/pytope/polytope.py#L601

    NOTE: This only requires vertex representations of P and Q, meaning that it will
    work even if ONE of P or Q is not full-dimensional (unbounded polyhedron).
    BUT the output must be a full-dimensional polytope!
    '''
    if P.vertices is None or Q.vertices is None:
        raise Exception('Polytopes must have a vertex representation for Minkowski Sum!')

    assert P.vertices.shape[1] == Q.vertices.shape[1], 'Polytopes must be of same dimension'
    n = P.vertices.shape[1]

    num_verts_P, num_verts_Q = (P.vertices.shape[0], Q.vertices.shape[0])
    msum_V = np.full((num_verts_P * num_verts_Q, n), np.nan, dtype=float)
    
    if num_verts_P <= num_verts_Q:
        for i in range(num_verts_P):
            msum_V[i*num_verts_Q:(i+1)*num_verts_Q, :] = Q.vertices + P.vertices[i, :].reshape(1, -1)
    else:
        for i in range(num_verts_Q):
            msum_V[i*num_verts_P:(i+1)*num_verts_P, :] = P.vertices + Q.vertices[i, :].reshape(1, -1)

    # result polytope as the convex hull of the pairwise sum of vertices
    out = qhull(msum_V)

    # check that the output is full-dimensional (bounded polyhedron)
    if len(out.b) == 0:
        raise Exception('Result of Minkowski Sum is not full-dimensional!')

    return out

def _pontryagin_difference(P: Polytope, Q: Polytope) -> Polytope:
    '''
    Pontryagin difference for two convex polytopes P and Q:
    P - Q = {x in R^n : x + q in P, for all q in Q}
    In halfspace representation, this is [P.A, P.b - Q.support(P.A)], with
    Q.support(P.A) a matrix in which row i is the support of Q at row i of P.A.

    Based on https://github.com/heirung/pytope/blob/master/pytope/polytope.py#L620

    NOTE: This requires halfspace representations of P and Q
    '''
    assert P.A.shape[1] == Q.A.shape[1], 'Polytopes must be of same dimension'
    m = P.A.shape[0]

    pdiff_b = np.full(m, np.nan)  # b vector in the Pontryagin difference P - Q
    # For each inequality i in P: subtract the support of Q in the direction P.A_i
    for i in range(m):
        ineq = P.A[i, :]
        pdiff_b[i] = P.b[i] - _support(Q, ineq)
        if pdiff_b[i] < 0:
            raise Exception('Result of Pontryagin Difference is invalid! Negative b value.')

    pdiff = Polytope(A=P.A.copy(), b=pdiff_b)

    # get a minimal representation of the result
    pdiff = _reduce(pdiff)

    return pdiff

def _projection(P: Polytope, dim:list, solver:str, abs_tol: float, verbose: int) -> Polytope:
    """This is just a wrapper around the polytope.projection function."""
    pc_P = projection(P, dim, solver, abs_tol, verbose)
    return Polytope(A=pc_P.A, b=pc_P.b, vertices=pc_P.vertices)

def _matrix_propagate_polytope(A:np.array, P: Polytope) -> Polytope:
    '''
    Propagate a polytope P through a matrix A.
    Based on propagating the vertices.
    '''
    assert P.vertices is not None, 'Polytope must have a vertex representation for propagation!'
    dim = P.vertices.shape[1]

    assert A.shape[1] == dim, 'A must have input dimension equal to {0}, the dimension of the polytope'.format(dim)

    verts = (A @ P.vertices.T).T
    return qhull(verts)

def _scale_polytope(a:float, P: Polytope) -> Polytope:
    '''
    Scale polytope P by float a.
    '''
    if not isinstance(a, (float, int)):
        raise NotImplementedError('Multiplier must be a float or int not {0}'.format(type(a)))
    
    if P.A.size != 0:
        return Polytope(P.A, a * P.b)
    else:
        print("[Warning] passed polytope has no H-Rep; returning scaled polytope only in V-Rep!")
        return Polytope(vertices=a * P.V)