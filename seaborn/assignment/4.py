import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

print(tips.head())

pivot_table = tips.pivot_table(index="day", columns="time", values="tip",aggfunc="mean")
sns.heatmap(
    pivot_table,
    annot=True,
    )
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.xlabel("Time of Day")
plt.ylabel("Day of Week")
plt.show()