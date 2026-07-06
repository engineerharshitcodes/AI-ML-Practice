import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset("tips")
print(tips.head())

sns.relplot(x="total_bill", y="tip", data=tips, hue="sex", markers=["o", "s"],style="smoker")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Total Bill vs Tip by Gender")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()

