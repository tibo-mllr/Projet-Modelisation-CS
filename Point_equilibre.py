from Simulateur_projet import *
from scipy.optimize import fsolve


def equilibre(p):
    c1, c2, c3, c4, c5, c6 = p

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


C_test = [0 for i in range(6)]
for i in range(6):
    C_test[i] = solution.y[i][-1]

c1, c2, c3, c4, c5, c6 = fsolve(equilibre, C_test)

print(c1, c2, c3, c4, c5, c6)
