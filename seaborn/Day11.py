import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

print(sns.get_dataset_names())

sns.load_dataset("tips")

rel_plot = sns.relplot(x="total_bill", y="tip", data=sns.load_dataset("tips"))