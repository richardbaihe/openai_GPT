#!/bin/bash
cur_dir=$(cd `dirname $0`; pwd)

if [ ! -e ${cur_dir}/../../data/temp ];then
	mkdir ${cur_dir}/../../data/temp
fi

cat ${cur_dir}/../../data/stop_seg_Ax.txt ${cur_dir}/../../data/stop_seg_Bx.txt > ${cur_dir}/../../data/temp/seg_ABx.txt &&
python ${cur_dir}/learn_bpe.py -i ${cur_dir}/../../data/temp/seg_ABx.txt -s 4000 -v -o ${cur_dir}/4000.codec &&
python ${cur_dir}/apply_bpe.py -i ${cur_dir}/../../data/temp/seg_ABx.txt -o ${cur_dir}/bpe_ABx.txt -c ${cur_dir}/4000.codec &&
./${cur_dir}/../Vocab/get_vocab.py < ${cur_dir}/bpe_ABx.txt > ${cur_dir}/vocab_bpe.txt &&
./${cur_dir}/../Vocab/resize_freq.py -i ${cur_dir}/vocab_bpe.txt -o ${cur_dir}/vocab.txt -s 10 &&
./${cur_dir}/../Vocab/unk.py ${cur_dir}/bpe_ABx.txt ${cur_dir}/vocab.txt ${cur_dir}/unk_ABx.txt &&
nums=$(cat unk_ABx.txt| wc -l)
let nums=nums/2
head -n ${nums} ${cur_dir}/unk_ABx.txt > ${cur_dir}/A.txt
tail -n ${nums} ${cur_dir}/unk_ABx.txt > ${cur_dir}/B.txt
paste -d '\t' ${cur_dir}/A.txt ${cur_dir}/B.txt > ${cur_dir}/bpe_unk_AB.tsv
paste -d '\t' ${cur_dir}/bpe_unk_AB.tsv ${cur_dir}/../../data/label.txt > ${cur_dir}/AB_unk.tsv
mv AB_unk.tsv ${cur_dir}/../../data/
mv vocab.txt ${cur_dir}/../../data/
rm ${cur_dir}/A.txt
rm ${cur_dir}/B.txt
rm ${cur_dir}/unk_ABx.txt
rm ${cur_dir}/vocab_bpe.txt
rm ${cur_dir}/bpe_ABx.txt
rm ${cur_dir}/bpe_unk_AB.tsv
