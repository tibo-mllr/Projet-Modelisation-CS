# installé avec la commande "pip install scipy"
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import numpy as np
import matplotlib.pyplot as plt

# Définitions des paramètres du système
a1, p1 = 0.7, 0.173
a2, p2 = 0.65, 0.231
a3, p3 = 0.65, 0.347
a4, p4 = 0.65, 0.693
a5, p5 = 0.55, 1.386
k = 1.6*10**(-10)
d = 0.3


# Définition des conditions initiales
c10 = 10**5
c20 = 10**6
c30 = 10**7
c40 = 0
c50 = 0
c60 = 0

# Définition du système d'équation différentielle


def signal(c6):
    return 1/(1+k*c6)


def systeme(t, X0):
    # Conditions initiales:
    c1 = X0[0]
    c2 = X0[1]
    c3 = X0[2]
    c4 = X0[3]
    c5 = X0[4]
    c6 = X0[5]

    # Système différentiel:
    dc1_dt = (2*a1*signal(c6)-1)*p1*signal(c6)*c1
    dc2_dt = (2*a2*signal(c6)-1)*p2*signal(c6)*c2 + \
        2*(1-a1*signal(c6))*p1*signal(c6)*c1
    dc3_dt = (2*a3*signal(c6)-1)*p3*signal(c6)*c3 + \
        2*(1-a2*signal(c6))*p2*signal(c6)*c2
    dc4_dt = (2*a4*signal(c6)-1)*p4*signal(c6)*c4 + \
        2*(1-a3*signal(c6))*p3*signal(c6)*c3
    dc5_dt = (2*a5*signal(c6)-1)*p5*signal(c6)*c5 + \
        2*(1-a4*signal(c6))*p4*signal(c6)*c4
    dc6_dt = 2*(1-a5*signal(c6))*p5*signal(c6)*c5 - d*c6

    return [dc1_dt, dc2_dt, dc3_dt, dc4_dt, dc5_dt, dc6_dt]


# Résolution
solution = solve_ivp(systeme, [0, 100], [
                     c10, c20, c30, c40, c50, c60], method='RK45', max_step=0.01)


# Vérification de la stabilité
#X_test = [318000, 3328000, 33226000, 249555000, 623886000, 2510000000]
#solution = solve_ivp(systeme, [0, 2000], X_test, method='RK45', max_step=0.01)

# Récupération des résultats
s = [signal(c6) for c6 in solution.y[5]]

if __name__ == "__main__":
    fig, axs = plt.subplots(4, 2)

    axs[0][0].plot(solution.t, solution.y[0], label="c1")
    axs[0][0].grid()
    axs[0][1].plot(solution.t, solution.y[1], label="c2")
    axs[0][1].grid()
    axs[1][0].plot(solution.t, solution.y[2], label="c3")
    axs[1][0].grid()
    axs[1][1].plot(solution.t, solution.y[3], label="c4")
    axs[1][1].grid()
    axs[2][0].plot(solution.t, solution.y[4], label="c5")
    axs[2][0].grid()
    axs[2][1].plot(solution.t, solution.y[5], label="c6")
    axs[2][1].grid()
    axs[3][0].plot(solution.t, s, label="Signal")

    plt.xlabel("Temps (jours)")
    plt.show()
