import pandas as pd

data = pd.read_csv("submission.csv")
# count number of negative values in the views column
print((data["views"] < 0).sum())
# print the index of the first 10 negative values
print(data[data["views"] < 0].index[:10])