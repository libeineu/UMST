# data_path下应该包含train.en train.de valid.en valid.de test.en test.de BPE后的结果
data_path="./data/wmt-en2de"
script_path="./scripts"
src="en"
tgt="de"
tag="wmt-en2de-group"

echo "==========生成intra group=========="
echo "......"
python3 ./get_intra_group.py $data_path
echo "==========生成intra group成功 ！=========="

echo "==========还原BPE=========="
echo "......"
sed "s/@@ //g" $data_path/train.$src > $data_path/train.$src.org
sed "s/@@ //g" $data_path/valid.$src > $data_path/valid.$src.org
sed "s/@@ //g" $data_path/test.$src > $data_path/test.$src.org


echo "==========生成inter group=========="
echo "......"
python3 ./get_inter_group.py $src $data_path


echo "==========解析inter group=========="
echo "......"
python3 $script_path/get_phrase.py $data_path/train.phrase.en $data_path/train.phrase.pos.en
python3 $script_path/get_phrase.py $data_path/valid.phrase.en $data_path/valid.phrase.pos.en
python3 $script_path/get_phrase.py $data_path/test.phrase.en $data_path/test.phrase.pos.en

if [ ! -d "data-bin" ]; then
        mkdir data-bin
fi

tmp="./data-bin/tmp"

if [ -d $tmp ]; then
        rm -rf $tmp
fi

mkdir $tmp

echo "==========生成translation data bin=========="
echo "......"
python3 preprocess.py --source-lang $src --target-lang $tgt --trainpref $data_path/train  --validpref $data_path/valid --testpref $data_path/test --destdir $tmp/translation-data  --workers 32 --joined-dictionary


echo "==========生成intra group data bin=========="
echo "......"
# train.en.tree
python3 $script_path/gen_dict.py 200 $data_path/dict.$src.mapping.txt
python3 preprocess.py --only-source --source-lang tree --trainpref $data_path/train.en  --validpref $data_path/valid.en --testpref $data_path/test.en --destdir $tmp/intra-data  --workers 32 --srcdict $data_path/dict.$src.mapping.txt


echo "==========生成inter group data bin=========="
echo "......"
# train.en.tree
python3 preprocess.py --only-source --source-lang $src --trainpref $data_path/train.phrase.pos  --validpref $data_path/valid.phrase.pos --testpref $data_path/test.phrase.pos --destdir $tmp/inter-data  --workers 32 --srcdict $data_path/dict.$src.mapping.txt


cd data-bin

mkdir $tag

cp tmp/translation-data/* $tag/

rename tree-None.tree $src-$tgt.mapping.$src tmp/intra-data/*
rename dict.tree.txt dict.tree.$src.txt tmp/intra-data/*
cp tmp/intra-data/* $tag

rename $src-None.$src $src-$tgt.phrase.$src tmp/inter-data/*
rename dict.$src.txt dict.phrase.$src.txt tmp/inter-data/*
cp tmp/inter-data/* $tag


echo "最终数据集处理完成！"


