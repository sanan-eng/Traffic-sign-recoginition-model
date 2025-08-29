import kagglehub

Download GTSRB dataset
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", path)
import pandas as pd
meta = pd.read_csv('Meta.csv')
print(meta.columns
      )