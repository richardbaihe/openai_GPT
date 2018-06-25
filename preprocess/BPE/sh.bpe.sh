if [ ! -e ../../data/temp ];then
	mkdir ../../data/temp
fi
cat ../../data/seg_Ax.txt ../../data/seg_Bx.txt > ../../data/temp/seg_ABx.txt &&
python learn_bpe.py -i ../../data/temp/seg_ABx.txt -s 4000 -v -o 4000.codec &&
python apply_bpe.py -i ../../data/temp/seg_ABx.txt -o bpe_ABx.txt -c 4000.codec &&
../Vocab/get_vocab.py < bpe_ABx.txt > vocab_bpe.txt &&
../Vocab/resize_freq.py -i vocab_bpe.txt -o vocab.txt -s 10 &&
../Vocab/unk.py bpe_ABx.txt vocab.txt unk_ABx.txt &&
nums=$(cat unk_ABx.txt| wc -l)
let nums=nums/2
head -n ${nums} unk_ABx.txt > A.txt
tail -n ${nums} unk_ABx.txt > B.txt
paste -d '\t' A.txt B.txt > bpe_unk_AB.txt
rm A.txt
rm B.txt
rm unk_ABx.txt
rm vocab_bpe.txt
rm bpe_ABx.txt
