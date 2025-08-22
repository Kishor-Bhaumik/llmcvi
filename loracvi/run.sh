# conda create -n cvi python=3.12
# conda activate cvi

# conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install faiss-gpu -c conda-forge

python cvi_qlora.py DEVICE=0 "+few_shots={0:50, 1:50, 2:50, 3:50}" RUN_NAME='fewshot50'
python cvi_qlora.py DEVICE=0 "+few_shots={0:30, 1:30, 2:30, 3:30}" RUN_NAME='fewshot30'
python cvi_qlora.py DEVICE=0 "+few_shots={0:10, 1:10, 2:10, 3:10}" RUN_NAME='fewshot10'


# python cvi_lgt.py DEVICE=1 TRAIN_SUBSET_RATIO=0.1 USE_SILHOUETTE=True RUN_NAME='w_cvi_0.1' BATCH_SIZE=16


# python cvi_lgt.py TRAIN_SUBSET_RATIO=0.1 USE_SILHOUETTE=True RUN_NAME='w_cvi_0.1' 
# python cvi_lgt.py TRAIN_SUBSET_RATIO=0.2 USE_SILHOUETTE=True RUN_NAME='w_cvi_0.2'
# python cvi_lgt.py TRAIN_SUBSET_RATIO=0.3 USE_SILHOUETTE=True RUN_NAME='w_cvi_0.3'
# python cvi_lgt.py TRAIN_SUBSET_RATIO=0.4 USE_SILHOUETTE=True RUN_NAME='w_cvi_0.4'
# python cvi_lgt.py TRAIN_SUBSET_RATIO=0.5 USE_SILHOUETTE=True RUN_NAME='w_cvi_0.5'


