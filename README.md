# Low-Rank Sinkhorn Factorization 
Code of the paper by Meyer Scetbon, Marco Cuturi and Gabriel Peyré

## A New Regularization Scheme for computing efficiently the Optimal Transport Cost
In this work, we propose to regularize the optimal transport (OT) problem by adding a low-rank constraint on the couplings. We propose an efficient algorithm to solve the problem and show on  several examples that our method outperforms the Sinkhorn algorithm in term of time-accuracy tradeoff. Our regularization can take advantage of the geometry of the problem, in particular when the cost matrix involved in the optimal transport problem admits a low-rank factorization. In this case, our method is able to compute the OT cost in linear time with respect to the number of samples. We present the time-accuracy tradeoff between different methods to compute the OT when the samples live on the unit sphere.
![figure](results/fig_acc.jpg)


This repository contains a Python implementation of the algorithms presented in the [paper](https://arxiv.org/pdf/2103.04737.pdf).
