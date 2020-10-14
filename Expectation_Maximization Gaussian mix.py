#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
from numpy.linalg import matrix_power


# In[39]:


image = mpimg.imread("im.jpg")
plt.imshow(image)
plt.show()

data = np.asarray(image)

print ("Image shape is: ", data.shape)
h = data.shape[0]
w = data.shape[1]
pixels = data.shape[0] * data.shape[1]
val = data.shape[2]

# NxD array
data = data.reshape((pixels,val))

print ("The total pixels are: ", pixels)

# we normalize the data
data = data / 255
print(data.shape)


# In[40]:


#initialize parameters
def initialize(D,K):
    
    #initialize π 
    p = np.full(K,1/K)
    print(p)

    #initialize μ 
    m = np.full((K,D),-1.0)
    for i in range (K):
        m[i,:] = np.random.uniform (0.1,0.9,D)
        
    #initialize Σ
    s = np.random.uniform (0.3, 0.7,K)
    
    m = np.asarray(m)
    s = np.asarray(s)
    
    return m, s, p


# In[41]:


def convergence(LLold,LLnew):
    conv = LLnew-LLold
    return conv


# In[42]:


def Expectation_maximazation(data,K):

    N,D = data.shape    
    #initialize parameters
    m,s,p = initialize(D,K)
    tolerance = 1e-6
    
    f, maximum = stability_f(data, m, s, p)
    for i in range(300):
        print("Iteration: ",i,"\t")
        
        #Expectation step
        LLold = LL(f,maximum)
        g = gamma(N,K,f,maximum)
        m,s,p = values(data,g)
        
        #Maximation step
        f, maximum = stability_f(data, m, s, p)
        LLnew = LL(f,maximum)
        
        conv = convergence(LLold,LLnew)
        
        if LLnew - LLold < 0:
            print ("Logarithmic likelyhood not decreasing!")
            exit()
        if np.abs(LLnew - LLold) < tolerance:
            print ("Reached tolerance for: ", K)
            
            return g, m
    
    return g,m
        
        


# In[43]:


def LL(f,maximum):
    temp = np.sum(np.exp(f-maximum[:,None]), axis=1)
    a = np.log(temp)
    return np.sum(maximum + a, axis=0)


# In[44]:


def gamma(N, K, f, maximum):
    for k in range(K):
        f[:,k] = f[:,k] - maximum
    gamma = np.zeros((N, K))
    f = np.exp(f)

    denominator = np.sum(f, axis=1)

    for k in range(K):
        gamma[:, k] = f[:, k] / denominator
    return gamma


# In[45]:


def stability_f(data, m, s, p):
    K = s.shape[0]
    f = np.zeros((data.shape[0], K))
    for k in range(K):
        pk = p[k]
        mk = m[k,:]
        sk = s[k]
        x_m = (data - mk)**2

        temp = np.sum((x_m / sk) + np.log(2*np.pi*sk), axis=1)
        f[:, k] = np.log(pk) - 0.5*temp
    # get the maximum of all f
    maximum = f.max(axis=1)
    return f, maximum


# In[46]:


def values(data, gamma):
    N = data.shape[0]
    K = gamma.shape[1]
    D = data.shape[1]
    m_new = np.zeros((K, D))
    p_new = np.zeros(K)
    s_new = np.zeros(K)
    g_sum = np.sum(gamma, axis=0)
    for k in range(K):

        gamma_k = gamma[:, k]
        # calculate the new m
        for d in range(D):
            m_new[k, d] = np.sum(data[:,d] * gamma_k, axis=0) / g_sum[k]

        # calculate the new sigma
        x_m = data - m_new[k, :]
        x_m = x_m**2
        x_m = np.sum(x_m, axis=1)

        s_new[k] = np.sum(gamma_k*x_m, axis=0) / (g_sum[k]*data.shape[1])

        # calculate the new p
        p_new[k] = g_sum[k]* (1/ N)
    return m_new, s_new, p_new


# In[47]:


def error(N, X_init, X_new):
    error = np.sum((X_init - X_new)**2)/N
    return error


# In[54]:


def run():
    k=[2,4,8,16,32,64]
  
    for i in k :
       
        g,m = Expectation_maximazation(data,i)
        
        #recreate image
        m = m*255
        m = m.astype(np.uint8)
        
        #new image
        new_img = np.zeros(data.shape)
        new_img = new_img.astype(np.uint8)
        
        #set pixel color 
        for p in range (data.shape[0]):
            temp = g[p].argmax()
            new_img[p] = m[temp]
            
        N = data.shape[0]
        X_init = data
        X_new = new_img/255
        
        #calculate error
        e = error(N, X_init, X_new)
        print("Error is :",e)
        
        new_img = new_img.reshape((h, w, 3))
        new_img = Image.fromarray(new_img, 'RGB')
        name = "new_image" + str(i)+".jpg"
        new_img.save(name)       
        


# In[53]:


run()


# In[ ]:




