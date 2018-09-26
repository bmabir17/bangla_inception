# bangla_inception
Bangla HandWritten Character Recognition using Convolutional Neural Network
## Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=Qlb5WslbaXM" target="_blank"><img src="http://img.youtube.com/vi/Qlb5WslbaXM/0.jpg" 
alt="MultiFaceTracker_demo1 Video" width="640" height="480" border="1" /></a>

## Requirements
  1. Tensorflow
  2. keras
  3. Flask
  4. matplotlib
  5. ipython (for training visiualization)

This project's front end is built using html, css and js. And i have used python flask framework as the server of the data. The front end takes the user input and gives it to the python server.
And for the recognition of the  characters i have trained a machine learning model using keras Library with python. The dataset used to train this recognition model is BanglaLekha Isolated.

# Related Publications
  <a href="https://doi.org/10.1007/978-981-13-0277-0_13"target="_blank">Published Paper on Springer</a>

## ToDO 
  1. Add a model training UI to the flask server. So that it will accpet a pickle file of training data and outputs the .ckpt     model
  2. Integrate the training data pre-prossing module into the UI, so that the user can generate and change the training data.
  3. 
