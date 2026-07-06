import matplotlib.pyplot as plt

days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

cities = ["New York", "London", "Delhi", "Tokyo"]

temperatures = [
    [22, 23, 21, 24, 25],   # New York
    [18, 19, 17, 20, 21],   # London
    [30, 32, 31, 33, 34],   # Delhi
    [25, 26, 24, 27, 28]    # Tokyo
]
fig,axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.plot(days, temperatures[i], marker='o')
    ax.set_title(cities[i])
        

plt.style.use('ggplot')
plt.style.available
plt.tight_layout(pad=2.0)
fig.suptitle("Temperature Trends in Different Cities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()