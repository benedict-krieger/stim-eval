import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-u','--user',help=f'User name, for which a sub-directory must exist')
parser.add_argument('-d','--data',help = f'Input tsv file')
args = parser.parse_args()

with open('llms.yaml','r') as f:
    llms = yaml.safe_load(f)

for llm in llms.keys():
    os.system(f'python llm-surprisal.py -u {args.user} -d {args.data} -m {llm}')

os.system('python merge_surprisal.py')