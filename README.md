# Learning Multiscale Transformer Models for Sequence Generation
This code is based on Fairseq v0.10.2
## Requirements and Installation
- PyTorch version >= 1.9.0
- python version >= 3.6
## Prepare Data
### Get Intra-group Relation
In order to get the Intra-group relation, you should first get a raw data file after BPE and run the following script:
```
python3 get_intra_group.py
```

### Get Inter-group Relation
In order to get the Inter-group relation, you should first get a raw data file. And install ```stanfordnlpCoreNLP``` software according to the steps of https://github.com/stanfordnlp/CoreNLP

Finally, run the following script on the raw data without bpe

```
python3 get_inter_group.py
```
## Train
### For WMT'14 En-De Task
```
python3 -u train.py data-bin/$data_dir
  --distributed-world-size 8 -s src -t tgt
  --task dp_tree_group_phrase_translation
  --arch phrase_transformer_t2t_wmt_en_de
  --optimizer adam --clip-norm 0.0
  --adam-betas '(0.9, 0.997)'
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 16000
  --lr 0.002 --min-lr 1e-09
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens 4096
  --update-freq 2
  --max-epoch 30
  --attention-dropout 0.1 -- relu-dropout 0.1
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --phrase
  --save-dir $save_dir
  --keep-last-epochs 10
```

### For Abstractive Summarization Task
```
python3 -u train.py data-bin/$data_dir
  --distributed-world-size 8 -s src -t tgt
  --task dp_tree_group_phrase_translation
  --arch phrase_transformer_t2t_wmt_en_de
  --share-all-embeddings
  --optimizer adam --clip-norm 0.0
  --adam-betas '(0.9, 0.997)'
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000
  --lr 0.002 --min-lr 1e-09
  --weight-decay 0.0001
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens 4096
  --update-freq 4
  --max-epoch 30
  --dropout 0.1 --attention-dropout 0.1 -- relu-dropout 0.1
  --truncate-source  --skip-invalid-size-inputs-valid-test --max-source-positions 500
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --phrase
  --save-dir $save_dir
  --keep-last-epochs 10
```

## Evaluation
### For WMT'14 En-De Task
```
python3 generate.py \
data-bin/wmt-en2de \
--task dp_tree_group_phrase_translation
--path $model_dir/$checkpoint \
--gen-subset test \
--batch-size 64 \
--beam 4 \
--lenpen 0.6 \
--output hypo.txt \
--quiet \
--remove-bpe
```

### For Abstractive Summarization Task

We use pyrouge as the scoring script. 

```
python3 generate.py \
data-bin/$data_dir \
--path $model_dir/$checkpoint \
--gen-subset test \
--truncate-source \
--batch-size 32 \
--lenpen 2.0 \
--min-len 55 \
--max-len-b 140 \
--max-source-positions 500 \
--beam 4 \
--no-repeat-ngram-size 3 \
--remove-bpe

python3 get_rouge.py --decodes_filename $model_dir/hypo.sorted.tok --targets_filename cnndm.test.target.tok
```
