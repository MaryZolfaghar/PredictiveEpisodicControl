# Temporal Abstraction in Episodic Control
Exploring the use of temporal abstraction in the context of episodic control.

### Environment Setup

Install [numpy](https://anaconda.org/conda-forge/numpy), [pytorch](https://anaconda.org/pytorch/pytorch), [matplotlib](https://anaconda.org/conda-forge/matplotlib), [networkx](https://anaconda.org/anaconda/networkx), [scikit-learn](https://anaconda.org/anaconda/scikit-learn)

Install jupyter-notebook (We have added a jupyter notebook and its corresponding Colab link).

### Usage
#### Random projection:
python train.py \
--embedding_type random \
--out_data_file ../results/MFEC/MFEC_rand_rooms_mnist_3knn.npy

#### VAE:
python train.py \
--embedding_type VAE \
--vae_batch_size 4 \
--vae_train_frames 100000 \
--vae_epochs 10 \
--lr 1e-5 \
--vae_print_every 100 \
--out_data_f

#### SR (DP):
python train.py \
--SR_gamma 0.99 \
--SR_batch_size 32 \
--SR_train_frames 1000000 \
--SR_epochs 10 \
--SR_train_algo DP \
--embedding_type SR \
--SR_embedding_type random \
--n_hidden 100 \
--lr 0.000006 \
--SR_filename ../results/MFEC_SR/random_DP_mnist_3knn \
--out_data_file ../results/MFEC_SR/MFEC_SR_rand_DP_rooms_mnist_3knn.npy

#### SR (TD):
python train.py \
--SR_gamma 0.99 \
--SR_batch_size 64 \
--SR_train_frames 1000000 \
--SR_epochs 200 \
--SR_train_algo TD \
--embedding_type SR \
--SR_embedding_type random \
--n_hidden 100 \
--lr 0.0001 \
--SR_filename ../results/MFEC_SR/random_TD_mnist_200epochs_3knn \
--out_data_file ../results/MFEC_SR/MFEC_SR_rand_TD_rooms_mnist_200epochs_3knn.npy
