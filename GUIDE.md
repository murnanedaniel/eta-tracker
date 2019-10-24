# Experiment Guide

This is a listing of the training and classification experiments run in this directory. We divide these by doublet classification and triplet classification.

## Doublet Training

| Size | Number of Graphs  | Training time  | Dataset dir  | Result dir  | Notes |
|------|---|---|---|---|---|
|  Large    | 32,768  |  6h13m | /doublet_data/hitgraphs_high_000  | /doublet_results/checkpoints_high/agnn001 |   |
|  Medium    | 2,000  |  31m | /doublet_data/hitgraphs_med_000  |  /doublet_results/agnn03 |   |
|      |   |   |   |   |   |


## Triplet Training

| Size | Number of Graphs  | Training time  | Dataset dir  | Result dir  | Notes |
|------|---|---|---|---|---|
|  Medium    | 1,920  |  54m | /triplet_data/hitgraphs_med  | /triplet_results/checkpoints_med/agnn01 |   |
|      | 1,920  |  1h12m | /triplet_data/hitgraphs_med  |  /triplet_results/checkpoints_med/agnn02 |   |
|      |   |   |   |   |   |


For future training: There is a strong link between the doublet trained results and the triplet trained results. Therefore the two sets should be closely linked by ID#. 
