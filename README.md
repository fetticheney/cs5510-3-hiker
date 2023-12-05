# About the Project
This application takes a 2D array of the strings of image filenames and returns the position in the array where it first successfully classifies a person in the image. In this case the definition of success is when the classification labels a bounding box with a confidence score of greater than 0.8. The pretrained model `fasterrcnn_resnet50_fpn` is used for classification.

### Dependencies
- `pip install opencv-python`\
Note: the packages below will take some time to install.
- `pip install torch`
- `pip install torchvision`

## Usage Instructions
1. Run the following command in the root folder of the project:\
`python3 find_hiker.py`
2. Output will first display contents of the `trail` array which contains the filenames of the image files (including the image of the hiker which was randomly placed in the array by the world.py script)
3. Application will then print to console the position in the array in which the hiker is found:\
example: `found hiker on map[ 0 , 2 ]`