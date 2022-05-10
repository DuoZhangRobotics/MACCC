import numpy as np

def fair(p, b):
    return np.dot(p, b) / (np.linalg.norm(p) * np.linalg.norm(b))

a=np.array([0.5,0.5])
b=np.array([0.75,0.25])
fair_loss = (fair(a,b) - 1) ** 2
print(fair_loss)