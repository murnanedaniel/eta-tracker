# Software environment setup for Cori Haswell and KNL
# Currently using my conda installation with pytorch-geometric
INSTALL_DIR=/global/cscratch1/sd/sfarrell/conda/pytorch-geometric
conda activate $INSTALL_DIR
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
