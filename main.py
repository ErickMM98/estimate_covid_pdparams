"""
Analizamos la paqueteríaa PDEParams.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import PDEparams as pde

data = pd.read_csv('CoV2019.csv')
china = data["China"][:]  # data["China"][:27]
days = data["Days"][:]
total = data["Total"][:]
deaths_china = data["Death China"][:]
other = data["Other"]
china_total = data["China"]
days_total = data["Days"]
deaths_china_total = data["Death China"]
deaths_outside_total = data["Death Outside"]

# Vamos a suponer que si estás en R si estás muerto.

N = 56 * 10 ** 3  # estimate of people affected by lock down

I = china.copy() / N
R = deaths_china_total.copy() / N
bool_array = (I <= 1)
I = I[bool_array]
R = R[bool_array]
S = 1 - I - R

init_I = I[0]
init_R = R[0]
init_S = S[0]


# fig, ax = plt.subplots()
# ax.plot(days[bool_array], S, color ='blue', marker = 's', alpha= 0.5,
#        label = 'Datos S')
# ax.plot(days[bool_array], I, color ='green',marker = 's',alpha= 0.5,
#        label = 'Datos I')
# ax.plot(days[bool_array], R, color ='red',marker = 's',alpha= 0.5,
#        label = 'Datos R')
# plt.show()

def SIRModel(x, t, b, g):
    '''The input z corresponds to the current state of the system, z = [s, i, r]. Since the input is in 1D, no
    pre-processing is needed.

    t is the current time.

    g and b correspond to the unknown parameters.
    '''
    s, i, r = x
    dzdy = [-b * s * i,
             b * s * i - g * i,
             g * i]
    return dzdy

def initial_s():
    return init_S

def initial_i():
    return init_I

def initial_r():
    return init_R

df_correct_data = pd.DataFrame({'t': days[bool_array].index / 27,
                                's': S,
                                'i': I,
                                'r': R})

bounds = [(0.5, 5),
          (0.5,5),
          (0.5,5)]

my_model = pde.PDEmodel(df_correct_data, SIRModel, [initial_r, initial_i, initial_r],
                        bounds=bounds, param_names=[r'$\beta$', r'$\gamma$'],
                        nvars=3, ndims=0, nreplicates=1)

#print(days[bool_array].index / 26)
print(my_model.fit())
#print(my_model.best_params)
#print(my_model.best_error)
#print(my_model.likelihood_profiles())
#my_model.plot_profiles()