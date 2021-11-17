import subprocess
import numpy as np
from matplotlib import pyplot as plt
import os

cmd = f'go run main.go'.replace('\\', '/')
print(cmd)

subprocess.check_output(cmd, shell=True)

data = np.genfromtxt('out.csv', delimiter=",")
print(data)


plt.plot(data[:, 0], data[:, 1], label="Track")
plt.plot(data[:, 2], data[:, 3], marker=".", label="Measure")
plt.plot(data[:, 4], data[:, 5], marker="*", label="UKF")
plt.title("Unscented Kalman Filter")
plt.legend(loc=2)
plt.tight_layout()
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.grid(True)

plt.show()


os.remove('out.csv')