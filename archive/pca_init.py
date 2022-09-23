import torch

# Given pytorch tensor X of shape (N, D), where D is the dimension
# of the data and N is the number of samples, return a pytorch tensor
# X_prime of shape (r, N), where we use SVD to project out machine
# precision principal components

# IMPUT: X is a pytorch tensor of shape (D, N)
# OUTPUT: X_new a pytorch tensor of shape (r, N), where r is the number
# of principal components with singular value greater than 1e-6

# U a pytorch tensor of shape (D, r), representing the principal
# components
def pca_init(X):
	U, S, V = torch.svd(X.T)

	S_r = torch.diag(S[S > 1e-6])
	r = S_r.shape[0]
	U_r = U[:,:r]
	V_r = V[:,:r]
	# new X of shape (r,N)
	# note this is essentially multiplying X on the left by U_r.T
	X_new = (S_r @ V_r.T).T
	return X_new, U_r
	