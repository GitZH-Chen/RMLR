"""
    Official Implementation of the SPD MLR presented in
    @inproceedings{chen2024rmlr,
        title={{RMLR}: Extending Multinomial Logistic Regression into General Geometries},
        author={Ziheng Chen and Yue Song and Rui Wang and Xiaojun Wu and Nicu Sebe},
        booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
        year={2024},
    }
"""

import math
import geoopt
import torch as th
import torch.nn as nn

from Geometry.spd.spd_matrices import SPDLogEuclideanMetric,SPDAdaptiveLogEuclideanMetric,SPDLogCholeskyMetric,\
    SPDAffineInvariantMetric,SPDBuresWassersteinMetric,SPDEuclideanMetric,\
    SPDPowerCholeskyMetric,SPDBuresWassersteinCholeskyMetric,\
    tril_param_metric,bi_param_metric,single_param_metric

class SPDRMLR(nn.Module):
    def __init__(self,n,c,metric='LEM',power=1.0,alpha=1.0,beta=0.):
        """
            Input X: (N,h,n,n) SPD matrices
            Output P: (N,dim_vec) vectors
            SPD parameter of size (c,n,n), where c denotes the number of classes
            Sym parameters (c,n,n)
        """
        super(__class__, self).__init__()
        self.n = n;self.c = c;
        self.metric = metric;self.power = power;self.alpha = alpha;self.beta = beta;

        self.P = geoopt.ManifoldParameter(th.empty(c, n, n), manifold=geoopt.manifolds.SymmetricPositiveDefinite())
        init_3Didentity(self.P)
        self.A = nn.Parameter(th.zeros_like(self.P))
        init_matrix_uniform(self.A, int(n ** 2))
        self.get()

        if self.metric=='ALEM':
            self.weight=nn.Parameter(th.ones(n))

    def forward(self,X):
        A_sym = symmetrize_by_tril(self.A)
        if self.metric=='ALEM':
            X_new = self.spd.RMLR(X, self.P, A_sym,self.weight)
        else:
            X_new = self.spd.RMLR(X, self.P, A_sym)

        return X_new

    def get(self):
        classes = {
            "LEM": SPDLogEuclideanMetric,
            "ALEM": SPDAdaptiveLogEuclideanMetric,
            "LCM": SPDLogCholeskyMetric,
            "AIM": SPDAffineInvariantMetric,
            "BWM": SPDBuresWassersteinMetric,
            "PEM": SPDEuclideanMetric,
            'PCM': SPDPowerCholeskyMetric,
            'BWCM': SPDBuresWassersteinCholeskyMetric
        }

        if self.metric in tril_param_metric:
            self.spd = classes[self.metric](n=self.n,alpha=self.alpha,beta=self.beta,power=self.power)
        elif self.metric in bi_param_metric:
            self.spd = classes[self.metric](n=self.n,alpha=self.alpha,beta=self.beta)
        elif self.metric in single_param_metric:
            self.spd = classes[self.metric](n=self.n,power=self.power)
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n},c={self.c},metric={self.metric},power={self.power},alpha={self.alpha},beta={self.beta})"

def symmetrize_by_tril(A):
    """"
    symmetrize A by the lower part of A, with [...,n,n]
    """
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    tmp_A_sym = str_tril_A + str_tril_A.transpose(-1, -2) + th.diag_embed(diag_A_vec)
    return tmp_A_sym

def init_matrix_uniform(A,fan_in,factor=6):
    bound = math.sqrt(factor / fan_in) if fan_in > 0 else 0
    nn.init.uniform_(A, -bound, bound)

def init_3Didentity(S):
    """ initializes to identity a (h,ni,no) 3D-SPDParameter"""
    h,n1,n2=S.shape
    for i in range(h):
        S.data[i] = th.eye(n1, n2)


class UnsqueezeLayer(nn.Module):
    def __init__(self, dim=1):
        super(__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Perform the unsqueeze operation
        return x.unsqueeze(self.dim)