# Grab AI for SEA Challenge
## Computer Vision
Model to automate the process of recognizing the details of the vehicles from images

## Dataset Source
https://ai.stanford.edu/~jkrause/cars/car_dataset.html

196 Car Labels
* Train Dataset = 8041
* Test Dataset = 8144

## Steps to train the model
1. Delete all contents in "model_inception" directory or place where the classification model store
2. From the meta information given by the dataset source, run the command
    python load_mat_file.py
3. Dump and segmentate based on bounding box value from meta information, run the command
    python prepareCarDataset.py
4. Run the following command to train the model
    python trainWithInception.py



## Steps to test the model

1. Run the following command. This will process in image and predict the car make and model

    python predict_car.py --file_to_test [filename] --bbox_x1 [bbox_value] --bbox_y1 [bbox_value] --bbox_x2 [bbox_value] --bbox_y2 [bbox_value]

## Author
Kevin