[<img src="https://img.shields.io/badge/arXiv-2409.19433-b31b1b"></img>](https://arxiv.org/abs/2409.19433)
[<img src="https://img.shields.io/badge/OpenReview|forum-lBp2cda7sp-8c1b13"></img>](https://openreview.net/forum?id=lBp2cda7sp)
[<img src="https://img.shields.io/badge/OpenReview|pdf-lBp2cda7sp-8c1b13"></img>](https://openreview.net/pdf?id=lBp2cda7sp)

# RMLR: Extending Multinomial Logistic Regression into General Geometries

This is the official code for our NeurIPS 2024 publication: *RMLR: Extending Multinomial Logistic Regression into General Geometries*. 

If you find this project helpful, please consider citing us as follows:

```bib
@inproceedings{chen2024rmlr,
    title={{RMLR}: Extending Multinomial Logistic Regression into General Geometries},
    author={Ziheng Chen and Yue Song and Rui Wang and Xiaojun Wu and Nicu Sebe},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
}
```
And our previous work on the flat SPD MLR, ALEM, and product Cholesky metrics (PCM and BWCM):
```bib
@inproceedings{chen2024spdmlr,
    title={Riemannian Multinomial Logistics Regression for {SPD} Neural Networks},
    author={Ziheng Chen and Yue Song and Gaowen Liu and Ramana Rao Kompella and Xiaojun Wu and Nicu Sebe},
    booktitle={Conference on Computer Vision and Pattern Recognition 2024},
    year={2024}
}
```
```bib
@article{chen2024adaptive,
  title={Adaptive Log-Euclidean metrics for {SPD} matrix learning},
  author={Chen, Ziheng and Song, Yue and Xu, Tianyang and Huang, Zhiwu and Wu, Xiao-Jun and Sebe, Nicu},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
```
```bib
@article{chen2024product,
  title={Product geometries on {Cholesky} manifolds with applications to {SPD} manifolds},
  author={Chen, Ziheng and Song, Yue and Wu, Xiao-Jun and Sebe, Nicu},
  journal={arXiv preprint arXiv:2407.02607},
  year={2024}
}
```

If you have any problem, do not hesitate to contact me ziheng_ch@163.com.

## Geometries
This source code contains eight SPD MLRs:
- "LEM": SPDLogEuclideanMetric,
- "ALEM": SPDAdaptiveLogEuclideanMetric,
- "LCM": SPDLogCholeskyMetric,
- "AIM": SPDAffineInvariantMetric,
- "BWM": SPDBuresWassersteinMetric,
- "PEM": SPDEuclideanMetric,
- "PCM": SPDPowerCholeskyMetric,
- "BWCM": SPDBuresWassersteinCholeskyMetric,

where ALEM comes from our TIP24, and PCM & BWCM come from our Arxiv24.

We also release the torch implementation for SO(3) in `Geometry.rotation`.

## Requirements

Install necessary dependencies by `conda`:

```setup
conda env create --file environment.yaml
```

**Note** that the [hydra](https://hydra.cc/) package is used to manage configuration files.

## Demo
 Demos of typical use can be found in `demo_spd.py`:
```python
import torch as th
from RieNets.spdnets.SPDMLR import SPDRMLR

# Set parameters
batch_size = 8  # Batch size
n = 5  # Dimension of SPD matrices
c = 3  # Number of classes

# Generate random SPD matrices of shape (batch_size, n, n)
X = th.randn(batch_size, 1, n, n)
X = X @ X.transpose(-1, -2)  # Ensure positive definiteness
X += th.eye(n) * 1e-3  # Add a small perturbation to guarantee strict positive definiteness

# Initialize the model
model = SPDRMLR(n=n, c=c, metric='LEM')

# Forward computation
output = model(X)

# Print results
print("Input X shape:", X.shape)
print("Output shape:", output.shape)
print("Output:", output)
```

## Experiments

The implementation of SPD MLR is based on our previous work:
    
- *Riemannian Multinomial Logistics Regression for SPD Neural Networks* [[CVPR 2024](https://arxiv.org/abs/2305.11288)] [[code](https://github.com/GitZH-Chen/SPDMLR)].

### Dataset

The preprocessed SPD from the HDM05 dataset can be found [[here](https://www.dropbox.com/scl/fi/x2ouxjwqj3zrb1idgkg2g/HDM05.zip?rlkey=4f90ktgzfz28x3i2i4ylu6dvu&st=oisp66vz&dl=0)].

Please download the datasets and put them in your folder.
If necessary, change the `path` accordingly in
`conf/RiemNets/dataset/HDM05_SO3.yaml` and `conf/RiemNets/dataset/HDM05_SPD.yaml`,

**Note:** Other datasets for the SPD networks can be found in our CVPR24 paper.

### Running experiments

To run experiments on the SPDNet (Tabs. 4.), run this command:

```train
bash exp_spdnets.sh
```

**Note:** You also can change the `hdm05_path` in `exp_spdnets.sh`, which will override the hydra config.



