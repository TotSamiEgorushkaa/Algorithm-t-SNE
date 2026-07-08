import numpy as np

def H_P_vectorized(D_matrix, beta_vector):
    P_matrix = np.exp(-D_matrix * beta_vector)
    
    np.fill_diagonal(P_matrix, 0)
    
    sum_P = np.sum(P_matrix, axis=1, keepdims=True)
    
    eps = 1e-12
    safe_sum_P = sum_P + eps
    
    H = np.log(safe_sum_P) + beta_vector * np.sum(D_matrix * P_matrix, axis=1, keepdims=True) / safe_sum_P
    P_matrix = P_matrix / safe_sum_P
    
    return H, P_matrix

def x2p(X, tol=1e-5, perplexity=30.0):
    n, d = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    
    np.fill_diagonal(D, 0)

    beta = np.ones((n, 1))       
    betamin = np.full((n, 1), -np.inf)
    betamax = np.full((n, 1), np.inf)
    logU = np.log(perplexity)
    
    for tries in range(50):
        H, P = H_P_vectorized(D, beta)
        

        Hdiff = H.reshape(-1, 1) - logU

        non_converged = np.abs(Hdiff) > tol

        if not np.any(non_converged):
            break
            
        case_greater = non_converged & (Hdiff > 0)
        betamin[case_greater] = beta[case_greater]

        inf_max = (betamax == np.inf) | (betamax == -np.inf)

        idx1 = case_greater & inf_max
        beta[idx1] *= 2.

        idx2 = case_greater & ~inf_max
        beta[idx2] = (beta[idx2] + betamax[idx2]) / 2.
        
        case_less = non_converged & (Hdiff <= 0)
        betamax[case_less] = beta[case_less]
        
        inf_min = (betamin == np.inf) | (betamin == -np.inf)

        idx3 = case_less & inf_min
        beta[idx3] /= 2.
        idx4 = case_less & ~inf_min
        beta[idx4] = (beta[idx4] + betamin[idx4]) / 2.

    print("Среднее значение sigma: %f" % np.mean(np.sqrt(1 / (2*beta))))
    
    return P

def pca(X, no_dims = 50):    

    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.dot(X_centered.T, X_centered)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    
    idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, idx]
    
    top_components = eigen_vectors[:, :no_dims]
    Y = np.dot(X_centered, top_components)
    
    return Y

def tsne(
    X, 
    no_dims = 2, 
    initial_dims = 50, 
    perplexity = 30.0,
    lr = 500.0,
    n_iter = 1000,
    early_exaggeration = 4.0,
    exaggeration_iters = 100
):
    
    X_reduced = pca(X, initial_dims).real
    n, _ = X_reduced.shape
    
    initial_momentum = 0.5
    final_momentum = 0.8
    min_gain = 0.01
    
    Y = np.random.randn(n, no_dims)
    iY = np.zeros((n, no_dims))  
    gains = np.ones((n, no_dims)) 
    
    P = x2p(X_reduced, tol=1e-5, perplexity=perplexity)
    P = P + P.T
    P = P / np.sum(P)
    P = P * early_exaggeration
    P = np.maximum(P, 1e-12)

    for it in range(n_iter):
        
        sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)
        num = -2.0 * np.dot(Y, Y.T) + sum_Y + sum_Y.T
        num = 1.0 / (1.0 + num)
        np.fill_diagonal(num, 0.0) 
        
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        stiffness = (P - Q) * num

        dY = 4.0 * (np.dot(np.diag(np.sum(stiffness, axis=1)), Y) - np.dot(stiffness, Y))

        momentum = initial_momentum if it < 20 else final_momentum

        directions_changed = (dY > 0.0) != (iY > 0.0)
        gains = (gains + 0.2) * directions_changed + (gains * 0.8) * (~directions_changed)
        gains = np.maximum(gains, min_gain)
        
        iY = momentum * iY - lr * (gains * dY)
        Y = Y + iY
        
        Y = Y - np.mean(Y, axis=0)
        
        if (it + 1) % n_iter == 0:
            kl_divergence = np.sum(P * np.log(P / Q))
            print(f"Итерация {it + 1}: значение функции потерь (KL) = {kl_divergence:.6f}")
            
        if it == exaggeration_iters:
            P = P / early_exaggeration
            
    return Y
