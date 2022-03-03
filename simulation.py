# %%

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import pandas as pd

# %%
#This block of code is needed to generate signals at random
def kernel(x1,x2, l = 10):                          
    return np.exp(-(x1 - x2)**2 /(2* l**2))

def make_signal(R=200, mean = 10):
    cov = np.zeros((R,R))
    for i in range(R):
        for j in range(i+1):
            cov[i,j] = cov[j,i] = kernel(i,j)
    fn = rand.multivariate_normal(mean + np.zeros(R), cov)
    return fn

def sigmoid(y):
    return 1/(1+np.exp(-y))
# %%
# This block of code has all the parameters for generating a data set
rand.seed(1234)

N = 2000                               # Number of data points
R = 200                                # Number of wavelenghths to measure at

sigma = 2                            # Variance of the random noise

n_continuos = 2                       # Number of interferaring signals that can take conrinuos values
continuos_correlation = [0.5, -0.2]            # How correlated each signal is with API fraction
continuos_sd = [1, 1]                # How strong each signal is 


n_discreat =  1                        # Number of interfearing signals that can take only descreat values
discreat_interaction = [10]
discreat_strength= [10]


mass_mean = 10
mass_var = 2                   # Amount the mass varies from one sample to the next
API_frac_mean = 0.1                    # Mean API fraction (must be less than 1)
API_frac_var = 0.0002                  # API fraction variance (set this very small)


if n_discreat != len(discreat_interaction) or n_discreat != len(discreat_strength):
    raise Exception("correlation with API  and signal strength must be specified for all descreat valued interfearing signals ")
if n_continuos != len(continuos_correlation) or n_continuos != len(continuos_sd):
    raise Exception("Correlation with API and variance must be specified for all continuos valued interfearing signals ")

# %%
API_signal = make_signal(R)
excipiant_signal = make_signal(R)

interfearing_cont = np.zeros((n_continuos, R))
interfearing_disc = np.zeros((n_discreat, R))

for i in range(n_continuos):
    interfearing_cont[i] = make_signal(R)
for i in range(n_discreat):
    interfearing_disc = make_signal(R)


plt.figure(figsize=(20,10))
plt.plot(API_signal, label = "API Signal")
plt.plot(excipiant_signal, label = "Exipiant Signal")
for i in range(n_continuos):
    plt.plot(interfearing_cont[i], label = "Conrinuos Interfearing signal %s" % i)
for i in range(n_discreat):
    plt.plot(interfearing_disc[i], label = "Descreat Interfearing signal %s" % i)
plt.xlabel("wavelength")
plt.ylabel("intensity")
plt.legend()
plt.show()
# %%

alpha = API_frac_mean**2*(1-API_frac_mean)/API_frac_var - API_frac_mean
beta = alpha*(1-API_frac_mean)/API_frac_mean

API_fraction = rand.beta(alpha,beta,N)


plt.figure()
plt.title("API fraction")
plt.hist(API_fraction, bins = 100)
plt.xlim(0,1)
plt.show()

# %%

mass = rand.normal(mass_mean,mass_var,N)

api = np.outer(mass*API_fraction,API_signal)
excipiant = np.outer(mass*(1-API_fraction),excipiant_signal)

continuos = np.zeros((N,R))
continuos_value = np.zeros((n_continuos, N))

discreat = np.zeros((N,R))
discreat_value = np.zeros((n_discreat,N))

noise = rand.normal(0,sigma, (N,R))

api_sd = np.sqrt(API_frac_var)


for i in range(n_continuos):
    cont_mean = continuos_sd[i]/api_sd *  continuos_correlation[i] *(API_fraction - API_frac_mean)
    cont_var = (1 - continuos_correlation[i]) * continuos_sd[i]
    continuos_value[i] = rand.normal(cont_mean, cont_var)
    continuos += np.outer(continuos_value[i], interfearing_cont[i]) 

for i in range(n_discreat):
    p = sigmoid((API_fraction - API_frac_mean)*discreat_interaction)
    discreat_value[i] = rand.binomial(1, p)
    discreat += discreat_strength[i]*np.outer(discreat_value[i], interfearing_disc[i])



# %%
data = api + excipiant + noise + continuos

plt.figure()
plt.title("observed data")
for i in range(N):
    plt.plot(data[i])
plt.show()

# %% 

df = pd.DataFrame(data)
df["API frac"] = API_fraction
df["mass"] = mass
for i in range(n_continuos):
    df["continuos interfearence %s" % str(i + 1)] = continuos_value[i]
for i in range(n_discreat):
    df["discreat interfearence %s" % str(i + 1)] = discreat_value[i]



df.head()

df.to_csv("data.csv")

# %%
