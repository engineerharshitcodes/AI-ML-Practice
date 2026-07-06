import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset("flights")

print(flights.head())

print(flights.pivot(index="month", columns="year", values="passengers"))

sns.heatmap(
    flights.pivot(index="month", columns="year", values="passengers"),
    cmap="coolwarm",
    annot=True,
    fmt="d"
)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()