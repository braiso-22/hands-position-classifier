# Hands Position Classifier

This is a Python project that trains a classifier model to predict which position your hand is in, based on input
images of your hand. The model can be used to classify your hand into one of several pre-defined positions,
such as open hand, closed fist, or pointing finger.

## Getting Started

To get started, you first need to clone the repository to your local machine.
You can do this by running this command:

```bash
$ git clone https://github.com/braiso-22/hands-position-classifier
$ cd hands-position-classifier
```

Then, you need to install the dependencies listed in the reqs.yaml file. 
You can do this by running this command:

```bash 
conda env create -f reqs.yaml hands-classifier
```

You can activate the environment by running this command:

```bash
conda activate hands-classifier
```

## Executing the program

To execute the program, you need to run the main.py file.

```bash
python main.py
```

### Menu options

#### Generate dataset

- This option asks you to enter the name of the position you want to generate the dataset for.
- Then opens a window where you can see your webcam feed.
- You can press the space bar to freeze the frame and press 'q' to exit the camera window.
- The generated dataset will be saved in the 'dataset' folder and in a subfolder named with the current date,
there are the images of your hand in the specified position and a csv file with the labels of the images.

#### Train model
    
- This option trains the model with the dataset in the 'dataset' folder.
- The model is only in the memory, so you need to save it if you want to use it in the next session.

#### Save model
    
- This option saves the model in the 'models' folder.
- The model is saved with the name you labeled it with.

#### Load model

- This option loads a model from the 'models' folder.
- You need to enter the name of the model you want to load.

#### Test model

- This option opens a window where you can see your webcam feed.
- You can press the space bar to freeze the frame and press 'q' to exit the camera window.
- The program will classify your hand into one of the pre-defined positions and display the result in the window.

#### Play
    
- This option opens a window where you can see your webcam feed and a firefox browser.
- First click the camera window and then click the start button in the browser.
- The model will classify your hand in real time and use it to control the game.

## Contributing
Contributions to this project are welcome. 
If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.