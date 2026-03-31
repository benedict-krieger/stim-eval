import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u','--user',help=f'User name, for which a sub-directory must exist')
parser.add_argument('-d','--data',help = f'Input csv file, tab separated')
args = parser.parse_args()

files = list(Path(f"{args.user}/results/").glob("*.csv"))
final_df = pd.read_csv(files[0], sep='\t')

for f in files[1:]:
    next_df = pd.read_csv(f, sep='\t')
    
    model_col = [c for c in next_df.columns if c.endswith("_surp")][0]
    
    join_keys = ['Item', 'Condition' ,'Word_Position', 'Word']
    
    final_df = final_df.merge(
        next_df[join_keys + [model_col]], 
        on=join_keys,
        how='left'
    )

print(f"Final Shape: {final_df.shape}")

filename = Path(args.data).stem
final_df.to_csv(f"{args.user}/results/{filename}_merged.csv", sep='\t', index=False)