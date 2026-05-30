# Real time had gesture detection: from rock paper scissors to sign interpretation

This project's goal is building a computer vision system capable of recognizing a variety of **hand signs and gestures** captured by a static camera. It uses a robust gesture recognition pipeline that can distinguish between different hand configurations in real time. 

As a proof-of-concept, the system is integrated into a game of **rock-paper-scissors**, where it detects each player's gesture and determines the outcome of each round. However, the underlying technology can be extended to recognize a wide range of hand signs, making it applicable for various applications such as sign language interpretation, human-computer interaction, and more.

The project is comprised of 2 parts: the model and the tracker (the latter serves also as demo)

## Installation and usage
You may want to use the `pip install -r requirements.txt` command to install all the packages you can use with their versions.

**Important**: for all the packages to work properly, you need to run the code with **Python 3.9**. We recommend using a virtual environment to manage the dependencies and avoid conflicts with other projects.

<!-- We remind you of the "requirements.txt" file with all the necessary programs and relative versions we used to run all the code  -->

    1. The model is trained from the merged_dataset.csv file that contains the hand landmarks of all the labeled gestures (rock, paper, scissors and thumbs up)
        To train the model simply run the "train_xgboost.py" file

    2. The tracker uses Mediapipe to recognize in real time the hand landmarks from the camera, and with the model classifies it as a gesture
        To run the demo simply execute "project.py"