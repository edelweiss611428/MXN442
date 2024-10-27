

L21_ZVCV = function(integrands, samples, grad, Q = 2){
  
  #ZVCV design matrix
  Z = getX(samples, grad, Q)
  
  #l21-norm regularisation
  glmnet_model = cv.glmnet(Z, integrands, family = "mgaussian", standardize.response = T) 
  
  #get regression coefficients
  beta = as.matrix(as.data.frame(lapply(coef(glmnet_model, s = glmnet_model$lambda.min), as.matrix)))[-1,]
  
  #CV expectation estimates 
  cv_integrands = integrands - Z%*%beta
  
  return(colMeans(cv_integrands))
  
}


L1_ZVCV = function(integrands, samples, grad, Q = 2){
  
  #ZVCV design matrix
  Z = getX(samples, grad, Q)
  
  #set-up
  ntasks = ncol(integrands)
  cv_estimates = numeric(ntasks)
  
  for(i in 1:ntasks){
    
    #l1-norm regularisation
    glmnet_model = cv.glmnet(Z, integrands[,i]) 
    #get regression coefficients
    beta = as.numeric(coef(glmnet_model, s = glmnet_model$lambda.min))[-1]
    cv_estimates[i] = mean(integrands[,i] - Z%*%beta)
    
  }
  
  return(cv_estimates)
  
}



L2_ZVCV = function(integrands, samples, grad, Q = 2){
  
  #ZVCV design matrix
  Z = getX(samples, grad, Q)
  
  #set-up
  ntasks = ncol(integrands)
  cv_estimates = numeric(ntasks)
  
  for(i in 1:ntasks){
    
    #l1-norm regularisation
    glmnet_model = cv.glmnet(Z, integrands[,i], alpha = 0) 
    #get regression coefficients
    beta = as.numeric(coef(glmnet_model, s = glmnet_model$lambda.min))[-1]
    cv_estimates[i] = mean(integrands[,i] - Z%*%beta)
    
  }
  
  return(cv_estimates)
  
}


