import numpy as np
import matplotlib.pyplot as plt


def fair(p, b):
    return np.dot(p, b) / (np.linalg.norm(p) * np.linalg.norm(b))

a=np.array([0.0,1.0])
b=np.array([0.75,0.25])
fair_loss = []
a_value=[]
for i in range(30):
    a[0] += 0.1
    fair_loss.append( (fair(a,b) - 1) ** 2)
    a_value.append(a[0])

plt.plot(a_value,fair_loss)
plt.xlabel("value of a[0]")
plt.xlabel("fair loss")
plt.savefig("tmp.png", dpi=500)