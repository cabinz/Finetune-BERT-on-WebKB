# Observation.

python bertclf-train.py --bert_name bert-base-uncased \
    --uni_lt cornell texas wisconsin washington misc \
    --max_len 512 --batch_siz 16 --lr 5e-6 --num_epoch 100
python huggingface-train.py bert_name prajjl1/bert-tiny \
    --uni_lt cornell texas wisconsin washington misc \
    --max_len 512 --batch_siz 16 --lr 5e-6 --num_epoch 20


# Tuning.
# python bertclf-train.py \
#     --uni_lt cornell texas wisconsin washington misc \
#     --max_len 512 --batch_siz 16 --lr 5e-6 --num_epoch 4
# python huggingface-train.py \
#     --uni_lt cornell texas wisconsin washington misc \
#     --max_len 512 --batch_siz 16 --lr 5e-6 --num_epoch 5