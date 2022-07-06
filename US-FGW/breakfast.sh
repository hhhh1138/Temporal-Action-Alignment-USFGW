python -m src.train \
    --data 'breakfast' --split 1 --gpu '0' \
    --model-type 'deterministic' --task-type 'set' --autoencoder-type 'MLP' --loss-type 'CrossEntropy' --predict-type 'OT'\
    --exp '' --sample_rate 15 --gamma 1.0 --beta 0.1 --b 0.1 --tau 0.1 \
    --badmm-loops 3000 --badmm-error-bound 0.005 --Lambdav 0.1 --Lambdaw 0 --z-dim 8 --lr 0.001 --enable-spectral 1 --enable-contrastive-learning 1
