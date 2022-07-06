python -m src.train \
    --data 'hollywood' --split 1 --gpu '0' \
    --model-type 'deterministic' --task-type 'set' --autoencoder-type 'MLP' --loss-type 'CrossEntropy' --predict-type 'OT'\
    --exp '' --sample_rate 15 --gamma 1.0 --beta 0.1 --b 0.1 --tau 0.01 \
    --badmm-loops 3000 --badmm-error-bound 0.001 --Lambdav 0 --Lambdaw 0 --z-dim 8 --lr 0.0003
