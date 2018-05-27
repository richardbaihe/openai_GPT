import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',required=True)
parser.add_argument('--output',required=True)

args = parser.parse_args()
in_path = args.input
out_path = args.output

test = pd.read_csv(in_path,sep='\t',header=None)
test[4] = 1
test.to_csv(out_path,index=None,header=None,sep='\t',columns=[0,4])
