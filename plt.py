import matplotlib.pyplot as plt
import numpy as np

# Use LaTeX rendering and set font to Times New Roman
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    "font.serif": ["Zen Maru Gothic"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 25,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16
})

mass_list = np.arange(5.204, 12.204 + 0.001, 0.2)
data = np.load('data.npz')
default_mass_dist = data['default_mass_dist']
adaptive_mass_dist = data['adaptive_mass_dist']
## calculate the variance in the two data and bar plot it
variance_default = np.var(default_mass_dist)
variance_adaptive = np.var(adaptive_mass_dist)  

plt.bar(['Vanilla CAJun', 'Adaptive CAJun'], [variance_default, variance_adaptive])
plt.xlabel('Method')
plt.ylabel('Variance')
plt.title('Variance of Distance Travelled with Mass Variations')
plt.show()

# plt.figure(figsize=(8, 6))  # Optional: Set larger figure size

# plt.plot(mass_list, default_mass_dist, label=r"\textbf{Vanilla CAJun}", linewidth=2)
# plt.plot(mass_list, adaptive_mass_dist + 3, label=r"\textbf{Adaptive CAJun}", linewidth=2)

# plt.xlabel("Mass")
# plt.ylabel("Distance")
# plt.title("Total Distance Travelled with Mass Variations")
# plt.legend()

# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.tight_layout()
plt.show()
