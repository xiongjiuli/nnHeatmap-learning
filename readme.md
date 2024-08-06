### nnHeatmap-learning

This network is derived from nnU-Net. Due to nnU-Net's effective self-configuration learning, we aimed to modify the task by replacing mask with heatmap. However, the network could not overfit on 1 or 20 data samples.

We modified the dataloader, loss function, and output format.

Just for recording the experience.
