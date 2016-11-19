# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:50:09 2016

@author: annawaldron
"""
import numpy as np
import math
from scipy.stats import beta
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


class MetropolisHastings(object):
    """Metropolis-Hastings sampling algorithm for data Y ~ Bin(n,theta) and 
    theta ~ Beta(a,b) with a Beta proposal distribution
    """

    def __init__(self, proposal_params, data, prior_params, iterations=10000, soln=[0,0]):
        """
        Parameters
        ----------
        proposal_params: list
            alpha and beta parameters of proposal Beta distribution
        data: int list 
            Y and n from binomial data (number of success and number of observations)
        prior_params: list
            a and b parameters of Beta prior on theta
        iterations: int
            how many times sampling will occur
        soln: list
            alpha and beta parameters of analytically derived Beta distribution
        """
        self.a = proposal_params[0]
        self.b = proposal_params[1]
        self.niter = iterations
        self.thetas = np.array([])
        self.acceptrate = 0
        self.data = data
        self.prior_params = prior_params
        self.soln = soln


    def nCr(self, n,r):
        """Computes n choose r (binomial coefficient)"""
        f = math.factorial
        return f(n) / f(r) / f(n-r)
        
        
    def beta_func(self, a, b):
        """Computes the beta function evaluated at integers a and b"""
        return self.gamma_func(a) * self.gamma_func(b) / self.gamma_func(a + b)
        
        
    def gamma_func(self, x):
        """Computes the gamma function of integer x"""
        if (x == 0.5):
            return np.sqrt(np.pi)
        if (x % 1 == 0 and x > 0):
            return math.factorial(x)
    
    
    def calculate_posterior(self, theta):
        """Calculates the posterior of theta with the given parameters"""
        Y = self.data[0] 
        n = self.data[1] 
        a = self.prior_params[0] 
        b = self.prior_params[1] 
        post_val = self.nCr(n, Y) * theta**(Y + a - 1) * (1 - theta)**(n - Y + b -1) / self.beta_func(a, b)
        return post_val
    
    
    def sample(self):
        """Samples from the posterior using M-H and calculates accept rate"""
        start_theta = np.random.beta(self.a, self.b) # start theta is sampled randomly
        self.thetas = np.append(self.thetas, start_theta) 
        accept_count = 0
        
        for i in range(0, self.niter):
            theta_t = self.thetas[i]
            theta_star = np.random.beta(self.a, self.b)
            pi_theta_star = self.calculate_posterior(theta_star)
            pi_theta_t = self.calculate_posterior(theta_t)
            q_theta_star = beta.pdf(theta_star, self.a, self.b)
            q_theta_t = beta.pdf(theta_t, self.a, self.b)
            val = pi_theta_star * q_theta_t / (pi_theta_t * q_theta_star)
            r = min(1, val)
            u = np.random.uniform(0, 1)
            if (u < r):
                new_theta = theta_star
                accept_count += 1
            else:
                new_theta = theta_t
            
            self.thetas = np.append(self.thetas, new_theta)
            self.accept_rate = accept_count / self.niter
        
        
    def traceplot(self):
        """Plots the traceplot of the samples"""
        x = np.array(range(0, self.niter + 1))
        plt.plot(x, self.thetas) 
        plt.title("Traceplot")
        plt.xlabel("Step #")
        plt.ylabel("theta")
    
    
    def hist_plot(self):
        """Plots the analytical soln over the histogram of the samples"""
        values = np.arange(0.1,1.0,0.01)
        y = beta.pdf(values, self.soln[0], self.soln[1])
        plt.figure()  
        plt.hist(self.thetas, bins=100, normed=True)  
        plt.plot(values, y, lw = 3.0, color = 'r')
        plt.title("Posterior: samples  and analytical soln")  
        plt.xlabel("theta")
        red_patch = mpatches.Patch(color='red', label='Analytical Solution')
        blue_patch = mpatches.Patch(color='blue', label='Sample Histogram')
        plt.legend(handles=[red_patch, blue_patch])

    

if __name__ == "__main__": 
    hw = MetropolisHastings([0.45, 0.45], [35, 100], [0.5, 0.5], 10000, [34.5, 65.5])
    hw.sample()
    hw.traceplot()
    hw.hist_plot()
    print("Accept rate = " + str(hw.accept_rate))
    
    
    
    
    
    
    
    
    
    
    