import numpy as np
import matplotlib.pyplot as plt

data = np.linspace(-70, 70, 300)
x = np.cos(np.deg2rad(data)) ** -2
plt.plot(data, x)
plt.xlabel(r'$\phi$ (degrees)')
plt.xlim(-80, 80)
plt.ylabel(r'$\sec^2(\phi)$')
plt.ylim(0, 8)
plt.grid()

plt.tight_layout()
plt.show()