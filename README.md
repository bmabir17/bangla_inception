# ReadMe
Bangla HandWritten Character Recognition using Convolutional Neural Network.
## Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=Qlb5WslbaXM" target="_blank"><img src="http://img.youtube.com/vi/Qlb5WslbaXM/0.jpg" 
alt="MultiFaceTracker_demo1 Video" width="640" height="480" border="1" /></a>

## Features
    1. Web UI for users to draw handwritten charecters and use that to predict which charecter the user has drawn

## Requirements
    1. Tensorflow
    2. keras
    3. Flask
    4. matplotlib
    5. ipython (for training visiualization and dataset pre-processing)

## Installation
    1. Need to download the dataset zip file ( Find the link below )
    2. Use the dataset to create a .pickle file which is used to train the Neural Network.
    3. Use the model.h5 and model.json and keep it in the /serve/model directory
    

# Related Publications
  <a href="https://doi.org/10.1007/978-981-13-0277-0_13" target="_blank">Published Paper on Springer</a>

## ToDO 
    1. Add a model training UI to the flask server. So that it will accpet a pickle file of training data and outputs the .ckpt model
    2. Integrate the training data pre-processing module into the UI, so that the user can generate and change the training data.
    3. API Documentation for Charecter recognition

## Contributions policy
    Any contribution to this project are welcome. If you want to contribute, please send a pull request.
    

  
## Dataset Used
### Banglalekha-isolated
  <a href="https://drive.google.com/file/d/0B23jlE3bGNoWODFyV0VpVVE5STA/view?usp=sharing" target="_blank">Download Link</a>

     Biswas M, Islam R, Shom GK, Shopon M, Mohammed N, Momen S, Abedin MA (2017) Banglalekha-isolated: a comprehensive bangla handwritten character dataset. arXiv:1703.10661
  
