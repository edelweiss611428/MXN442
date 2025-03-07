# Multitask Zero-Variance Control Variates via L2,1-Norm Regularisation

Constructing Zero-Variance Control Variates following the schemes in Mira et al. (2013), South et al. (2023) is equivalent to solving a linear regression problem, which easily becomes over-parameterised in high-dimensional applications. Penalised regression techniques such as LASSO and ridge regression can be used to estimate the regression coefficients. 

Often, there are more than one (T) expectation term to be estimated, and these tasks might be related. However, the previously mentioned ZVCV methods solve the T regression problems separately (i.e., in a single-task fashion). We have proposed a multitask regularised ZVCV framework based on ℓ2,1-norm regularisation, which may help capture the task relatedness by enforcing group sparcity. It implicitly assumes that the T tasks are related. We expect that the statistical efficiency of the single-task regularised ZVCV methods will be improved in a multi-task learning framework.

This Github repository stores the code, data, and results of the study, submitted as part of the MXN442 project. Multiple seeds were set to make the results in this project reproducible. Note that it could take quite a long time to run the code.

### References

[Mira, A., Solgi, R. and Imparato, D., 2013. Zero variance markov chain monte carlo for bayesian estimators. Statistics and Computing, 23, pp.653-662.](https://link.springer.com/article/10.1007/s11222-012-9344-6)

[South, L.F., Oates, C.J., Mira, A. and Drovandi, C., 2018. Regularized Zero-Variance Control Variates. arXiv preprint arXiv:1811.05073.](https://arxiv.org/abs/1811.05073)

[Argyriou, A., Evgeniou, T. and Pontil, M., 2008. Convex multi-task feature learning. Machine learning, 73, pp.243-272.](https://link.springer.com/article/10.1007/s10994-007-5040-8)

### Usage

The repository hosts an R project, consisting of five main working folders:

- code: where we store the code of the involved methods regularisedZVCV.R) and the experiments (experiment1.R, experiment2.R, experiment3.R) in the simulation study.
Users should be able to run the code without additional effort as all required libraries have been included in the renv folder. If that does not work, they could install the required packages listed on requiredPackages.R. The only additional requirement is the R version (R 4.4.1). Once you run each experimentID.R file, it will fetch the data/models from the corresponding files in data (dir) and model (dir) and save the results to results (dir). 
- renv: where we store the installed package.
- data: where we store the required data in json format.
- model: where we store Stan code for the models in the simulation study.
- results: where we store the simulation study results for the three experiments in the simulation study, including the Stan models, runtimes, and estimated values. Note that the results for the experiment1 are not available at 11:26 27/10/2024 due to being deleted accidentally (rm(list = ls()) and will be recovered soon.

