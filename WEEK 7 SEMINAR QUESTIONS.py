#!/usr/bin/env python
# coding: utf-8

# # week 7 Seminar Questions 

# In[5]:


import numpy as np
import pandas as pd
import os
#using os we don't need to use %d for percentages 


#  BINOMIAL TREES 
# 

# QUESTION 1 

# In[6]:



# 5 imputs for European Call Option 

S0 = 10                 # spot stock price
K = 11                  # strike
T = 0.25                # maturity 
r = 0.04                # risk free rate 
sigma = 0.25            # volatility (diffusion coefficient), we calibrate volatility, not extimate
N = 5                   # number of time steps (number of periods)
payoff = "call"         # payoff

#after we run it, Python record the data 


# In[7]:


dT = float(T) / N                             # Delta t
u = np.exp(sigma * np.sqrt(dT))               # up factor
d = 1.0 / u                                   # down factor


# In[8]:


# we start to create the tree 


# In[9]:



S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[11]:


S


# In[12]:


# the results is 6 values because it is a 5 time steps tree


# In[13]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
p


# In[14]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))  

#create a loop 

if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V

#we only have 2 values 


# In[15]:



# CODE for European Option
# create a loop inside the loop 
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[17]:


print('European ' + payoff + ' is', str( V[0,0]))


# QUESTION 2

# Same tree as before, we just need to change the payoff from call to put

# In[18]:


payoff = "put"


# In[19]:



S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# In[20]:


# CODE for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[22]:


print('European ' + payoff + ' is', str( V[0,0]))


# QUESTION 3 

# Use American Option, using the same data as before. 
# 
# American option can be more expensive than european, because it happens before. 

# In[24]:


# CODE for American Option
if payoff =="call":
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            V[i,j] = np.maximum(S[i,j] - K,np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1]))
elif payoff =="put":
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            V[i,j] = np.maximum(K - S[i,j],np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1]))
V


# In[26]:


print('American ' + payoff + ' is', str( V[0,0]))


# indeed, American Option IS MORE EXPENSIVE. 

# QUESTION 4 

# Montecarlo simulation on 1 call 

# In[29]:


def mcs_simulation_np(m,n):       #m is the number of steps and n is the number of simulation

    #n = 10000 times for the asset price with M = 90 steps
    
    M = m
    I = n
    
# Define the variables 

    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t]) 
    return S


# In[28]:


S = mcs_simulation_np(90,10000)


# In[31]:


import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=S[-1,:], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-preiod values')


# QUESTION 5 

# European Call 

# In[35]:


p = np.mean(np.maximum(S[-1,:] - K,0))
print('European call', str(p))

#this will cause a large error because the time frame is too small 


# QUESTION 6

# Binary Call 

# In[38]:


cp = (S[-1,:]  - K)>0 #Boolean, we add >0 to set up a boolean to check the value is true, then it change to an antigen
bpc = np.mean(np.maximum(cp.astype(int),0))
print('Binary call', str(bpc))


# Binary option for Put

# In[39]:


pp = (K - S[-1,:])>0
bpp = np.mean(np.maximum(pp.astype(int),0))
print('Binary put', str(bpp))


# Binary Put-Call parity

# In[41]:


bpc + bpp


# In[ ]:




