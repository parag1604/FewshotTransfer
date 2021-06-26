# README #

This is folder contains scripts for prior-based segmentation approaches, and sample data with slices stored in the axial view.

There are two approaches:
1. Contour Propagation CNN
2. Guided Pix2pix

SaveSlices.py is the script to preprocess the volumes and save slices as separate images.

### Environment setup instructions ###

Install dependencies from the environement.yml file into the virtual/conda environment.

(or)

Install the following dependencies in a virtual/conda environment:

* tensorflow v2.0
* cudatoolkit (appropriate version according to your driver)
* medpy
* albumentations

### Preprocessing instructions ###

Sample data is provided in the data folder.
The SaveSlices.py has the instructions for reading volumes, resampling and saving slices.

Directory Structure:
```

+-- axial
|   +-- Scripts
|   |   +-- datagen.py : Defines the data loader class for training with priors. 
|   |			 Loads data slice-wise, and automatically resizes. Also selects slices for training with partial annotations.
|   |	+-- models.py : Defines the unet, unet with bi-directional feature transforms, discriminator and guided gan functions
|   |	+-- utils.py : Defines helper functions
|   |	+-- predict.py : Reads a volume and selects annotated slices as priors for predicting segmentation.
|   |	+-- SaveSlices_gen.py : Preprocesses the volumes and stores each slice as a separate file (for faster training)
|   |	+-- train.py : Defines the train functions for the guided pix2pix and contour propagation and runs the main function. (Example given below)
|   +-- Data:
|   |	+-- Original_Vols : CT scans of the spine
|   |	|   +-- Image 
|   |   |       +-- <vol_name>.<ext>
|   |   |   +-- Mask
|   |   |       +-- <vol_name>_seg.<ext>
|   |   +-- Resampled_Vols : CT volumes after resampling to a uniform voxel spacing
|   |   |   +-- Image 
|   |	|       +-- <vol_id>_vols.<ext>
|   |   |    +-- Mask
|   |   |       +-- <vol_id>_vols.<ext>
|   |	+-- Axial_Slices : Slices stored as separate images
|   |	|   +-- Lumbar
|   |	|   |	+-- Image
|   |   |   |   |   +-- <vol_id>_slice<slice_no.>.<ext>
|   |   |   |   +-- Mask 
|   |   |   |   |   +-- <vol_id>_slice<slice_no.>.<ext>
|   |	|   +-- Whole
|   |	|   |	+-- Image
|   |   |   |   |   +-- <vol_id>_slice<slice_no.>.<ext>
|   |   |   |   +-- Mask 
|   |   |   |   |   +-- <vol_id>_slice<slice_no.>.<ext>
|   |   |   +-- Rest
|   |	|   |	+-- Image
|   |   |   |   |   +-- <vol_id>_slice<slice_no.>.<ext>
|   |   |   |   +-- Mask 
|   |   |   |   |   +-- <vol_id>_slice<slice_no.>.<ext>
|   +-- README.md

```


### Execution ###

General execution format:

Execute train.py with the following command line arguments:

or

Modify arguments in args_train.json and run train_json.py without any command line arguments.

Command Line Arguments:
```  
'--model' : *ContourPropagation or *GuidedPix2Pix
'--save_path_top' :  Name of the output folder
'--save_path_add' :  Name of the folder to save weights for the model being trained
'--data_path' :  Path to training data containing 'Image' and 'Mask' folders
'--val_path' : Path to validation data containing 'Image' and 'Mask' folders
'--load_pretrained' : boolean, False by default. Set to True if loading pretrained weights
'--pretrained_disc_weight_path' : Path to pre-trained weights for the discriminator - guided pix2pix
'--pretrained_gen_weight_path' : Path to pre-trained weights for the generator - guided pix2pix
'--pretrained_cpp_weight_path' : Path to pre-trained weights for the contour propagatoion cnn
'--epochs' : Number of epochs to train for
'--batch_size': Default 5
'--lrg' : Discriminator lr, default=0.0002
'--lrd' : Generator lr,  default=0.0002
'--lrcpp' : Contour Propagation CNN lr, default = 0.0001
'--crop_height': input-size default=256
'--crop_width': input-size default=256
'--validation': boolean, To perform validation or not
'--front_buffer' : Number of slices to remove from the beginning of each volume
'--activation' : 'tanh' or 'sigmoid', or any other activation function recognised by keras
'--segmentation_loss' : The loss for training the unet
'--discriminator_loss' : The loss for training the discriminator 
'--activation' : 'tanh' or 'sigmoid', or any other activation function recognised by keras
'--image_guide': Using the segmentation mask as input and image as guide
'--mask_guide' : Using the segmentation mask as the guide, image as input
'--q': distance of the prior
'--train_with_sparse' : boolean, True if training is to be done with sparse annotations
'--method': 'Random' , 'Central' , 'Front' - for choosing the location of the annotated slices in the sparsely annotated volume
'--percentage' : percentage of annotated slices in each volume
'--perc_sparse_vol' : percentage of total training volumes which are assumed to have sparse annotations

```

For example, to train the Contour Propagation CNN execute:
```
python train.py --model 'ContourPropagation' --q 3 --save_path_add 'CPP_q3' --data_path <path_to_training_data> --val_path <path_to_validation_data> --validation True
                --epochs 100 --front_buffer 3 --activation 'sigmoid' --train_with_sparse True --method 'Random' --percentage 20 --perc_sparse_vol 100

```
To train the Guided Pix2Pix execute:

```
python train.py --model 'GuidedPix2Pix' --q 3 --save_path_add 'GPP_q3' --data_path <path_to_training_data> --val_path <path_to_validation_data> --validation True
                --epochs 100 --front_buffer 3 --activation 'tanh'--train_with_sparse True --method 'Random' --percentage 20 --perc_sparse_vol 100 --mask_guide True
                --lrg 0.00002


```


