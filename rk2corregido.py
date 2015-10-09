#! /usr/bin/env python

'''
Este script resuelve el pendulo simple usando RK2.
'''

import numpy as np
import matplotlib.pyplot as plt


A = np.pi / 30
w1 = np.sqrt(10)

plt.figure(1)
plt.clf()

t = np.linspace(0, 5 * 2 * np.pi / w1, 400)

plt.plot(t, A * np.cos(w1 * t),'r',label="teorico")  # ERROR! ya que phi inicial es A, no podia ser seno.


def f(phi, w):
    return w, -10 * np.sin(phi)

def get_k1(phi_n, w_n, h, f):
    f_eval = f(phi_n, w_n)
    return h * f_eval[0], h * f_eval[1]

def get_k2(phi_n, w_n, h, f):
    k1 = get_k1(phi_n, w_n, h, f)
    f_eval = f(phi_n + k1[0]/2, w_n + k1[1]/2)
    return k1,(h * f_eval[0], h * f_eval[1]) # ERROR! faltaba retornar k1 para ocuparlo en rk2_step, en la parte de phi_n1 y w_n1 que requerian de k1.

def rk2_step(phi_n, w_n, h, f):
    k1,k2 = get_k2(phi_n, w_n, h, f)
    phi_n1 = phi_n + (k2[0]+k1[0])/2   # ERROR! en esta parte se habia definido mal phi_n1 y w_n1
    w_n1 = w_n + (k2[1]+k1[1])/2
    return phi_n1, w_n1

N_steps = 40000
h = 10. / N_steps
phi = np.zeros(N_steps)
w = np.zeros(N_steps)

phi[0] = A   # ERROR! dada esta definicion es que se tuvo que cambiar a coseno el grafico inicial
w[0] = 0
for i in range(1, N_steps):
    phi[i], w[i] = rk2_step(phi[i-1], w[i-1], h, f)



t_rk = [h * i for i in range(N_steps)]

plt.plot(t_rk, phi, 'g',label='metodo rk2')



plt.title('phi vs tiempo ',fontsize=14)
plt.xlabel('$\ tiempo[seg] $')
plt.ylabel('$\phi(t) [rad] $', fontsize=13)
plt.legend()
plt.show()
plt.draw()
