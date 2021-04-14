import numpy as np
import scipy.linalg as splin
import sys

m = 2**int(sys.argv[1])
n = int(sys.argv[2]) 

#Here is were we generate the Gaussian Matrix M
M = np.random.normal(0, 1, size=(m, n))

#We then perform SVD
SVD = np.linalg.svd(M, full_matrices = False)
U = SVD[0]
VT = SVD[2] 

#This step generates the gaussian vector b
b = np.random.normal(0, 1, size= m)

#This step replaces the eigenvalue of M with the uniform eigenvalues (1,10^6)
S = np.random.uniform(0, 10**6, n)
S[0] = 1
S[n-1] = 10**6

#This step creates the desired A matrix
A = np.matmul(np.matmul(U,np.diag(S)),VT)

#Lines 27-57 is where the SHRT projection algorithm is performed
#Given a desired precision epsilon we compute c
epsilon = float(sys.argv[3])
c = int(np.floor(n*np.log(m*n)/epsilon))

print("m = ", m, "n=", n, "epsilon = ", epsilon, "c = ",c )

#These next few steps produce the random projection by selecting 
#a random column in each row to be the nonzero entry
P = np.zeros((c,m))
for i in range(c):
	j = np.random.randint(0,m)
	P[i,j] = np.sqrt(m/c)


#We produce an m x m hadamard 
H = splin.hadamard(m)

#Here we produce the matrix D by randomly flipping signs on the diagonal
#of an identity matrix
D = np.zeros(m)
for i in range(m):
	D[i] = (-1)**np.random.randint(0,2)
D = np.diag(D)

#For simplicity we just compute the PHD matrix directly
#instead of using the optimal SHRT algorithm
PHD = np.matmul(np.matmul(P,H),D)

#This is the output of the SHRT alg
x_output = np.matmul(np.matmul(np.linalg.pinv(np.matmul(PHD,A)),PHD),b)
Ax_b_output = np.matmul(A,x_output) - b

#This is the output of the deterministic algorithm
#I just directly use numpy linear system solver
x_opt = np.linalg.lstsq(A,b, rcond = None)[0]
Ax_b_opt = np.matmul(A,x_opt) - b

#This computes the 2-norms of the errors
E1 = np.linalg.norm(Ax_b_output,2)
E2 = np.linalg.norm(Ax_b_opt,2)

#This computes the relative errors
rel_app_err = E1/E2

print("relative approximation error =")
print(rel_app_err)
