import numpy as np

# Placeholder values for Sigma (variance-covariance matrix), mu (expected returns), and T (liquidity vector)
Sigma = np.array([[1,2,3],[4,0,6],[7,8,9]])  # Fill in with actual variance-covariance matrix
mu = np.array([1,2,3])       # Fill in with actual expected return vector
T = np.array([4,5,6])        # Fill in with actual liquidity vector
u = 2                   # Value of z2 (expected return)
l = 2                    # Value of z3 (liquidity)
one=np.ones(len(mu))

# Calculating the inverse of Sigma
Sigma_inv = np.linalg.inv(Sigma)

# Define the C matrix
a = np.transpose(mu) @ Sigma_inv @ mu
b =  np.transpose(mu)@ Sigma_inv @ T
c = np.transpose(mu) @ Sigma_inv @ one
d = np.transpose(T) @ Sigma_inv @ T
e = np.transpose(T) @ Sigma_inv @ one
f = np.transpose(one) @ Sigma_inv @ one
C = np.array([[a, b, c], [b, d, e], [c, d, f]])

detC=np.linalg.det(C)

# Calculate C inverse
C_inv = np.linalg.inv(C)

#front multiplier
aa=Sigma_inv @ mu
bb=Sigma_inv @ T
cc=Sigma_inv @ one 


# Calculate vectors x0, d2, and d3
x0 = (1/detC)*((b*e-c*d)*aa+(b*c-a*e)*bb+(a*d-b*b)*cc)
d2 = (1/detC)*((d*f-e*e)*aa+(c*e-b*f)*bb+(b*e-c*d)*cc)
d3=(1/detC)*((c*e-b*f)*aa+(a*f-c*c)*bb+(b*c-a*e)*cc)

# Finally, calculate x
x = x0 + u * d2 + l * d3

# Output x
print(x)







