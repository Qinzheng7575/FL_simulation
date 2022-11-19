from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
three_bit = [10.0, 12.51, 16.93, 19.07, 20.47, 21.39, 21.49, 21.75, 21.99,
             22.17, 22.8, 23.43, 23.16, 23.08, 23.2, 22.96, 23.39, 23.0, 22.8, 23.36]
four_bit = [10.12, 23.03, 26.89, 35.98, 39.16, 40.0, 41.01, 42.04, 43.27,
            43.73, 44.31, 44.56, 45.24, 45.5, 45.55, 45.32, 45.28, 45.23, 45.31, 45.21]
eight_bit = [10.93, 21.12, 29.98, 34.01, 37.58, 39.67, 41.44, 42.47, 42.65,
             44.1, 44.93, 45.04, 45.53, 45.71, 45.35, 46.03, 46.88, 47.24, 47.49, 47.77]
full_bit = [10.32, 22.58, 28.36, 33.48, 36.3, 39.25, 40.87, 42.15, 44.09,
            45.12, 45.8, 46.29, 46.87, 47.46, 48.01, 48.37, 48.68, 48.45, 49.57, 49.33]

# x = range(20)
# plt.xticks(np.arange(0, 22, 2))
# plt.plot(x, three_bit, color='black', linestyle='--', label='3bit')
# plt.plot(x, four_bit, label='4bit')
# plt.plot(x, eight_bit, label='8bit')
# plt.plot(x, full_bit, color='black', linestyle='--', label='full')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('accuracy/%')
# plt.xticks(np.arange(0, 22, 2))
# plt.grid(linestyle='--')
# plt.show()
bad_scene = [10.0, 11.89, 16.44, 20.14, 21.31, 21.57, 22.38, 23.38, 25.22,
             25.2, 27.34, 29.6, 33.96, 34.8, 37.22, 37.47, 39.26, 40.33, 40.11, 40.12]
good_scene = [11.57, 12.14, 15.34, 20.1, 24.04, 24.05, 22.01, 25.33, 28.11,
              31.53, 34.52, 36.69, 38.89, 37.89, 39.41, 41.33, 41.17, 42.24, 43.23, 43.74]
x = range(10)
plt.plot(x, three_bit[10:], color='black', linestyle='--', label='3bit')
plt.plot(x, bad_scene[10:], label='bad_scene')
plt.plot(x, good_scene[10:], label='good_scene')
plt.plot(x, full_bit[10:], color='black', linestyle='--', label='full')
plt.legend(loc=2)
plt.xlabel('epoch')
plt.ylabel('accuracy/%')
plt.xticks(np.arange(0, 12, 2))
plt.grid(linestyle='--')
plt.show()
