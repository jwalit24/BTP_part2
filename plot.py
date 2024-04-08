import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_positive_definite(Sigma):
    return np.all(np.linalg.eigvals(Sigma) > 0)

def calculate_portfolio_weights(Sigma, mu, T, u, l):
    one=np.ones(len(mu))
    if is_positive_definite(Sigma):
        Sigma_inv = np.linalg.inv(Sigma)
        a = np.transpose(mu) @ Sigma_inv @ mu
        b =  np.transpose(mu)@ Sigma_inv @ T
        c = np.transpose(mu) @ Sigma_inv @ one
        d = np.transpose(T) @ Sigma_inv @ T
        e = np.transpose(T) @ Sigma_inv @ one
        f = np.transpose(one) @ Sigma_inv @ one
        C = np.array([[a, b, c], [b, d, e], [c, e, f]])

        detC=np.linalg.det(C)

        C_inv = np.linalg.inv(C)

        aa=Sigma_inv @ mu
        bb=Sigma_inv @ T
        cc=Sigma_inv @ one 

        x0 = (1/detC)*((b*e-c*d)*aa+(b*c-a*e)*bb+(a*d-b*b)*cc)
        d2 = (1/detC)*((d*f-e*e)*aa+(c*e-b*f)*bb+(b*e-c*d)*cc)
        d3=(1/detC)*((c*e-b*f)*aa+(a*f-c*c)*bb+(b*c-a*e)*cc)


        # print(x0)
        # print(d2)
        # print(d3)

        x = x0 + u * d2 + l * d3

        # Output x
        # print(x)
        # print("Sum of weights: ", np.sum(x))
        return x
    else:
        print("Matrix is not positive definite:\n")



def plot_3d_returns_variance_esg(portfolio_returns, portfolio_variances, portfolio_esgs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(portfolio_variances, portfolio_returns, portfolio_esgs, c='r', marker='o')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Returns')
    ax.set_zlabel('ESG Score')
    plt.show()


def generate_plot(Sigma, mu, ESG_scores, u_values, l_values):
    portfolio_returns = []
    portfolio_variances = []
    portfolio_esgs = []
    for u in u_values:
        for l in l_values:
            weights = calculate_portfolio_weights(Sigma, mu, ESG_scores, u, l)
            portfolio_return = np.dot(weights.T, mu)
            portfolio_variance = np.dot(weights.T, np.dot(Sigma, weights))
            portfolio_esg = np.dot(weights.T, ESG_scores)
            portfolio_returns.append(portfolio_return)
            portfolio_variances.append(portfolio_variance)
            portfolio_esgs.append(portfolio_esg)
    
    plot_3d_returns_variance_esg(portfolio_returns, portfolio_variances, portfolio_esgs)


# if __main__ == "__main__":
#     Sigma = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
#     ESG_scores = np.array([70, 85, 90])
#     mu = np.array([0.12,0.23,0.06])

#     u_values = np.linspace(0, 0.3, 100)
#     l_values = np.linspace(50, 100, 100)

#     generate_plot(Sigma, mu, ESG_scores, u_values, l_values)
