#! /usr/bin/env python
import argparse,sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="learn BPE-based word segmentation")

parser.add_argument('--input','-i',type=argparse.FileType('r',encoding='utf-8'),default=sys.stdin,metavar='PATH',help="Input text (default: standard input).")
parser.add_argument('--output', '-o', type=argparse.FileType('w',encoding='utf-8'), default=sys.stdout, metavar='PATH',help="Output file for BPE codes (default: standard output)")
parser.add_argument('--symbols', '-s', type=int, default=10000,help="Create this many new symbols (each representing acharacter n-gram) (default: %(default)s))")

args = parser.parse_args()
all_lines = args.input.readlines()
args.output.write('UNK\n')
for line in all_lines:
    num = int(line.split()[-1])
    if num < args.symbols:
        break
    args.output.write(' '.join(line.split()[:-1]) + '\n')

