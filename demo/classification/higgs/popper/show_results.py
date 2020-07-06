#/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

list_reports = glob.glob("results/*/report.csv")
dir_list = glob.glob("results/*")
list_reports.sort()
dir_list.sort()

results = pd.read_csv(list_reports[-1], sep=",")

sns.barplot(x = 'library', y = 'time', data = results)
plt.title('Performance of the libraries with HIGGS dataset')
plt.savefig(dir_list[-1] + "/report.png")
plt.show()
