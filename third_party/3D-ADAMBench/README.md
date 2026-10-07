# 3D-ADAMBench

This repository includes the code used in benchmarking the leading model in 3D and RGB+3D Anomaly Detection and Localisation on the novel 3D-ADAM Dataset.

The model selected for this is [TransFusion](https://github.com/maticfuc/eccv_transfusion), to which we make minor adjustments to handle our dataset.These include adding the 3D-ADAM dataset as an option in the dataloader, adding our categories as a dictionary for the arguments in Experiment.py and updating the model to handle 320 x 256 pixel images, as opposed to the intended 256 x 256 utilised by the MVTec3D-AD it previously trained on.

This repository also includes the guide produced for the Industry Expert Survey to provide instruction for labelling volunteers at the task of annotating defects from the dataset.

The 3D-ADAM dataset can be found publicly available in our [HuggingFace](https://huggingface.co/datasets/pmchard/3D-ADAM) repository.