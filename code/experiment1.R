# rm(list = ls())
source("./code/requiredPackages.R")
source("./code/regularisedZVCV.R")



mesquite_data = fromJSON(file="./data/mesquite.json")
mes_mod = stan_model(file = "./models/mesquite.stan")

nexps = 100
niter = 1250
nwarmup = 1000
nchain = 1
d = 8


l21zv1 = l21zv2 = l21zv3 = l21zv4 = l21zv5 = matrix(0,100,d)
lzv1 = lzv2 = lzv3 = lzv4 = lzv5 = matrix(0,100,d)
rzv1 = rzv2 = rzv3 = rzv4 = rzv5 = matrix(0,100,d)
zv1 = zv2 = zv3 = zv4 = zv5 = matrix(0,100,d)
cf =  matrix(0,100,d)
vanilla = matrix(0,100,d)


t_l21zv1 = t_l21zv2 = t_l21zv3 = t_l21zv4 = t_l21zv5 = numeric(100)
t_lzv1 = t_lzv2 = t_lzv3 = t_lzv4 = t_lzv5 = numeric(100)
t_rzv1 = t_rzv2 = t_rzv3 = t_rzv4 = t_rzv5 = numeric(100)
t_zv1 = t_zv2 = t_zv3 = t_zv4 = t_zv5= numeric(100)
t_cf = numeric(100)
t_vanilla = numeric(100)



for(i in 1:nexps){
  print(i)
  set.seed(i)
  
  t1 = proc.time()[3]
  
  stan_fit = sampling(mes_mod,data=mesquite_data,iter=niter, chains = nchain,
                      warmup = nwarmup)
  
  
  sample_ = rstan::extract(stan_fit)
  params = cbind(beta = sample_$beta, sigma = sample_$sigma)
  
  t2 = proc.time()[3]
  
  vanilla[i,] = colMeans(params)
  t_vanilla[i] = t2-t1
  
  
  stan_fit_name = paste0("./results/experiment1/stan_fit/stan_fit_",i,".RData")
  save(stan_fit, file = stan_fit_name)
  
  pars <- as.matrix(stan_fit)
  skel <- get_inits(stan_fit)[[1]]
  
  grad <- t(apply(params, MARGIN = 1, FUN = function(theta) {
    grad_log_prob(stan_fit, upars = unconstrain_pars(stan_fit, relist(theta, skel)), 
                  adjust_transform = T)
  }))
  
  unconstrain_pars = t(apply(params, MARGIN = 1, FUN = function(theta) {
    unconstrain_pars(stan_fit, relist(theta, skel))
  }))
  

  #ZVCV
  print("ZVCV")
  #ZV1
  
  t1 = proc.time()[3]
  
  ZVCV::zvcv(params, unconstrain_pars, grad, options = list(polyorder = 1,
                                                            regul_reg = F))$expectation -> zv1[i,]
  
  t2 = proc.time()[3]
  t_zv1[i] = t2-t1
  
  #ZV2
  
  t1 = proc.time()[3]
  
  ZVCV::zvcv(params, unconstrain_pars, grad, options = list(polyorder = 2,
                                                            regul_reg = F))$expectation -> zv2[i,]
  
  t2 = proc.time()[3]
  t_zv2[i] = t2-t1
  
  #ZV3
  
  t1 = proc.time()[3]
  
  ZVCV::zvcv(params, unconstrain_pars, grad, options = list(polyorder = 3,
                                                            regul_reg = F))$expectation -> zv3[i,]
  
  t2 = proc.time()[3]
  t_zv3[i] = t2-t1
  
  #LASSO
  print("LASSO")
  #lZV1
  
  t1 = proc.time()[3]
  
  lzv1[i,] = L1_ZVCV(params, unconstrain_pars, grad, 1)
  
  t2 = proc.time()[3]
  t_lzv1[i] = t2-t1
  
  #lZV2
  
  t1 = proc.time()[3]
  
  lzv2[i,] = L1_ZVCV(params, unconstrain_pars, grad, 2)
  
  t2 = proc.time()[3]
  t_lzv2[i] = t2-t1
  
  #lZV3
  
  t1 = proc.time()[3]
  
  lzv3[i,] = L1_ZVCV(params, unconstrain_pars, grad, 3)
  
  t2 = proc.time()[3]
  t_lzv3[i] = t2-t1
  
  #lZV4
  
  t1 = proc.time()[3]
  
  lzv4[i,] = L1_ZVCV(params, unconstrain_pars, grad, 4)
  
  t2 = proc.time()[3]
  t_lzv4[i] = t2-t1
  
  
  #lZV5
  
  t1 = proc.time()[3]
  
  lzv5[i,] = L1_ZVCV(params, unconstrain_pars, grad, 5)
  
  t2 = proc.time()[3]
  t_lzv5[i] = t2-t1
  
  
  #Ridge
  print("Ridge")
  #rZV1
  
  t1 = proc.time()[3]
  
  rzv1[i,] = L2_ZVCV(params, unconstrain_pars, grad, 1)
  
  t2 = proc.time()[3]
  t_rzv1[i] = t2-t1
  
  #rZV2
  
  t1 = proc.time()[3]
  
  rzv2[i,] = L2_ZVCV(params, unconstrain_pars, grad, 2)
  
  t2 = proc.time()[3]
  t_rzv2[i] = t2-t1
  
  #rZV3
  
  t1 = proc.time()[3]
  
  rzv3[i,] = L2_ZVCV(params, unconstrain_pars, grad, 3)
  
  t2 = proc.time()[3]
  t_rzv3[i] = t2-t1
  
  #rZV4
  
  t1 = proc.time()[3]
  
  rzv4[i,] = L2_ZVCV(params, unconstrain_pars, grad, 4)
  
  t2 = proc.time()[3]
  t_rzv4[i] = t2-t1
  
  #rZV5
  
  t1 = proc.time()[3]
  
  rzv5[i,] = L2_ZVCV(params, unconstrain_pars, grad, 5)
  
  t2 = proc.time()[3]
  t_rzv5[i] = t2-t1
  
  #L21
  print("L21")
  #l21ZV1
  
  t1 = proc.time()[3]
  
  l21zv1[i,] = L21_ZVCV(params, unconstrain_pars, grad, 1)
  
  t2 = proc.time()[3]
  t_l21zv1[i] = t2-t1
  
  #l21ZV2
  
  t1 = proc.time()[3]
  
  l21zv2[i,] = L21_ZVCV(params, unconstrain_pars, grad, 2)
  
  t2 = proc.time()[3]
  t_l21zv2[i] = t2-t1
  
  #l21ZV3
  
  t1 = proc.time()[3]
  
  l21zv3[i,] = L21_ZVCV(params, unconstrain_pars, grad, 3)
  
  t2 = proc.time()[3]
  t_l21zv3[i] = t2-t1
  
  #l21ZV4
  
  t1 = proc.time()[3]
  
  l21zv4[i,] = L21_ZVCV(params, unconstrain_pars, grad, 4)
  
  t2 = proc.time()[3]
  t_l21zv4[i] = t2-t1  
  
  
  #l21ZV5
  
  t1 = proc.time()[3]
  
  l21zv5[i,] = L21_ZVCV(params, unconstrain_pars, grad, 5)
  
  t2 = proc.time()[3]
  t_l21zv5[i] = t2-t1  
  
  #CF
  print("CF")
  
  t1 = proc.time()[3]
  cf[i,] = CF(params, unconstrain_pars, grad)$expectation
  t2 = proc.time()[3]
  t_cf[i] = t2-t1 
  
  
  
}


estimate_results = list(l21zv1 = l21zv1, l21zv2 = l21zv2, l21zv3 = l21zv3, l21zv4 = l21zv4, l21zv5 = l21zv5,
                        lzv1 = lzv1, lzv2 = lzv2, lzv3 = lzv3, lzv4 = lzv4, lzv5 = lzv5,
                        rzv1 = rzv1, rzv2 = rzv2, rzv3 = rzv3, rzv4 = rzv4, rzv5 = rzv5,
                        zv1 = zv1, zv2 = zv2, zv3 = zv3, 
                        cf = cf, vanilla = vanilla)

runtime_results = data.frame(t_l21zv1, t_l21zv2, t_l21zv3, t_l21zv4, t_l21zv5,
                             t_lzv1, t_lzv2, t_lzv3, t_lzv4, t_lzv5,
                             t_rzv1, t_rzv2, t_rzv3, t_rzv4, t_rzv5,
                             t_zv1, t_zv2, t_zv3,
                             t_cf, t_vanilla)

vrf_results = data.frame(l21zv1 = apply(l21zv1, 2, var), l21zv2 = apply(l21zv2, 2, var), l21zv3 = apply(l21zv3, 2, var), l21zv4 = apply(l21zv4, 2, var), l21zv5 = apply(l21zv5, 2, var),
                         lzv1 = apply(lzv1, 2, var), lzv2 = apply(lzv2, 2, var), lzv3 = apply(lzv3, 2, var), lzv4 = apply(lzv4, 2,var), lzv5 = apply(lzv5, 2, var),
                         rzv1 = apply(rzv1, 2, var), rzv2 = apply(rzv2, 2, var), rzv3 = apply(rzv3, 2, var), rzv4 = apply(rzv4, 2,var), rzv5 = apply(rzv5, 2, var),
                         zv1 = apply(zv1, 2, var), zv2 = apply(zv2, 2, var), zv3 = apply(zv3, 2, var),
                         cf = apply(cf,2, var), vanilla = apply(vanilla,2,var))


runtime_ratio = c(mean((t_l21zv1+t_vanilla)/t_vanilla),
                  mean((t_l21zv2+t_vanilla)/t_vanilla),
                  mean((t_l21zv3+t_vanilla)/t_vanilla),
                  mean((t_l21zv4+t_vanilla)/t_vanilla),
                  mean((t_l21zv5+t_vanilla)/t_vanilla),
                  mean((t_lzv1+t_vanilla)/t_vanilla),
                  mean((t_lzv2+t_vanilla)/t_vanilla),
                  mean((t_lzv3+t_vanilla)/t_vanilla),
                  mean((t_lzv4+t_vanilla)/t_vanilla),
                  mean((t_lzv5+t_vanilla)/t_vanilla),
                  mean((t_rzv1+t_vanilla)/t_vanilla),
                  mean((t_rzv2+t_vanilla)/t_vanilla),
                  mean((t_rzv3+t_vanilla)/t_vanilla),
                  mean((t_rzv4+t_vanilla)/t_vanilla),
                  mean((t_rzv5+t_vanilla)/t_vanilla),
                  mean((t_zv1+t_vanilla)/t_vanilla),
                  mean((t_zv2+t_vanilla)/t_vanilla),
                  mean((t_zv3+t_vanilla)/t_vanilla),
                  mean((t_cf+ t_vanilla)/t_vanilla),
                  mean(t_vanilla/t_vanilla))

nmethods = ncol(vrf_results)
VRF = colMeans(vrf_results[,nmethods]/vrf_results)
comp_eff = 1/runtime_ratio * VRF

methods = c("l21zv1", "l21zv2", "l21zv3", "l21zv4", "l21zv5", "lzv1", "lzv2", "lzv3", "lzv4", "lzv5",
            "rzv1", "rzv2", "rzv3", "rzv4", "rzv5", "zv1", "zv2", "zv3", "cf", "vanilla")
round(VRF,3)
round(comp_eff,3) 

agg_results = data.frame(
  vrf = round(VRF,3),
  comp_eff = round(comp_eff,3))

save(estimate_results, file = "./results/experiment1/estimate_results.RData")
save(runtime_results, file = "./results/experiment1/runtime_results.RData")
save(vrf_results, file = "./results/experiment1/vrf_results.RData")
save(agg_results, file = "./results/experiment1/overall_results.RData")

xtable(agg_results)



