# Software environment setup for Cori GPU
# Currently using my conda installation with pytorch-geometric
INSTALL_DIR=/global/cscratch1/sd/sfarrell/conda/pytorch-geometric-gpu
conda activate $INSTALL_DIR
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
