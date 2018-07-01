#!/usr/bin/env bash
#!/bin/bash
cur_dir=$(cd `dirname $0`; pwd)

cat ${cur_dir}/../../data/char_Ax.txt ${cur_dir}/../../data/char_Bx.txt > ${cur_dir}/char_ABx.txt &&
python ${cur_dir}/../Vocab/get_vocab.py < ${cur_dir}/char_ABx.txt > ${cur_dir}/vocab_char.txt &&
python ${cur_dir}/../Vocab/resize_freq.py -i ${cur_dir}/vocab_char.txt -o ${cur_dir}/vocab.txt -s 10 &&
python ${cur_dir}/../Vocab/unk.py ${cur_dir}/char_ABx.txt ${cur_dir}/vocab.txt ${cur_dir}/unk_ABx.txt &&
nums=$(cat unk_ABx.txt| wc -l)
let nums=nums/2
head -n ${nums} ${cur_dir}/unk_ABx.txt > ${cur_dir}/A.txt
tail -n ${nums} ${cur_dir}/unk_ABx.txt > ${cur_dir}/B.txt
paste -d '\t' ${cur_dir}/A.txt ${cur_dir}/B.txt > ${cur_dir}/char_unk_AB.tsv
paste -d '\t' ${cur_dir}/char_unk_AB.tsv ${cur_dir}/../../data/label.txt > ${cur_dir}/char_AB_unk.tsv
mv char_AB_unk.tsv ${cur_dir}/../../data/
mv vocab.txt ${cur_dir}/../../data/char_vocab.txt
rm ${cur_dir}/A.txt
rm ${cur_dir}/B.txt
rm ${cur_dir}/unk_ABx.txt
rm ${cur_dir}/vocab_char.txt
rm ${cur_dir}/char_ABx.txt
rm ${cur_dir}/char_unk_AB.tsv
