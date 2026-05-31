# Real time had gesture detection: from rock paper scissors to sign interpretation

This project's goal is building a computer vision system capable of recognizing a variety of **hand signs and gestures** captured by a static camera. It uses a robust gesture recognition pipeline that can distinguish between different hand configurations in real time. 

As a proof-of-concept, the system is integrated into a game of **rock-paper-scissors**, where it detects each player's gesture and determines the outcome of each round. However, the underlying technology can be extended to recognize a wide range of hand signs, making it applicable for various applications such as sign language interpretation, human-computer interaction, and more.

The project is comprised of 2 parts: the model pipeline and the real-time tracker integrated within the game (the latter serves also as demo)

## Installation and usage
You may want to use the `pip install -r requirements.txt` command to install all the packages you require with their versions.

> **Important**: for all the packages to work properly, you need to run the code with **Python 3.9**. We recommend using a virtual environment to manage the dependencies and avoid conflicts with other projects.

### Real-time tracker and game demo
To run the real-time tracker integrated with the rock-paper-scissors game, simply execute the `project.py` file. This will open a window with the camera feed, where you can play against another player. You will be assigned a player number (Player 1 or Player 2) depending on which side of the camera you are on. The game will keep track of the score and display it on the screen.

To begin a round, both players should hold the "thumbs-up" gesture in front of the camera for a few seconds to signal that they are ready. When the round starts, the players have three seconds (indicated both on the screen and via a sound cue) to show their gesture (rock, paper, or scissors) in front of the camera. The system will then detect the gestures, determine the winner of the round, and update the score accordingly. The game will continue indefinitly until you close the window, which you may do by pressing the `escape` key.

The tracker draws a bounding box around the detected hand and displays the predicted gesture label on the screen along with a confidence score. 

> **Note**: the game logic expects two players (i.e., two hands) to be detected in each round. Thus, having more than two hands in the camera feed (either the players' other hands, or the hands of other people in the feed) may cause unexpected behavior. We recommend ensuring that only the two players' playing hands are visible in the camera feed for the best experience.

### Model pipeline
You can find the pre-trained model and the label encoder in the "model" folder (`rps_model.pkl` and `rps_label`, respectively). However, if you want to train the model on your own dataset, you may run `extract_landmarks.py` to extract the hand landmarks from **labeled images** of the gestures you want to recognize, and then run `train_xgboost.py` to train the model on the extracted landmarks.

> **Note**: you *may* run the game-integrated tracker with your own model, but it expects a set of specific gestures (rock, paper, scissors and thumbs up) to work properly. If you want to use it with a different set of gestures, you will need to modify the code accordingly.

Otherwise, here is the general workflow we followed to build the model and the tracker:
1. **Collecting the data**: we used a mixture of landmark and hand gesture datasets available online (the first from Zenodo, available [here](https://zenodo.org/records/18108472?preview_file=hand-gestures.csv); the latter from Kaggle, available [here](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)). To extract the hand landmarks from the images, we used the `extract_landmarks.py` script, which uses Mediapipe to detect the hand landmarks and save them in a CSV file. We then merged the extracted landmarks with the original labels to create a final dataset (`merged_dataset.csv`) that contains both the landmarks and the corresponding gesture labels. We do this via the `merge_datasets.py` script.
2. **Data augmentation**: to increase the robustness of the model and prevent overfitting, we applied data augmentation techniques to the extracted landmarks. This included adding random noise, applying random rotations, and scaling the landmarks. We also **dropped** the z-axis of the landmarks, as it was not necessary for our task and could introduce noise into the model.
3. **Feature engineering**: we performed feature engineering on the extracted landmarks to create new features that could help the model better distinguish between different gestures. This included calculating the distances between specific landmarks, such as the tips of the fingers and the wrist, and the distance between all fingers.
4. **Model training**: we trained an XGBoost classifier and a Random Forest classifier on the extracted landmarks and the engineered features. While both models performed well, we chose to use the XGBoost model for the real-time tracker due to its slightly better performance.
5. **Game logic and real-time tracking**: we integrated the trained model into a real-time tracker that uses Mediapipe to detect the hand landmarks from the camera feed. The tracker then uses the trained model to classify the detected landmarks into one of the predefined gestures (rock, paper, scissors, or thumbs up). The game logic is implemented to determine the winner of each round based on the detected gestures.


<!-- We remind you of the "requirements.txt" file with all the necessary programs and relative versions we used to run all the code  -->

<!-- 1. The model is trained from the merged_dataset.csv file that contains the hand landmarks of all the labeled gestures (rock, paper, scissors and thumbs up)
    To train the model simply run the "train_xgboost.py" file

2. The tracker uses Mediapipe to recognize in real time the hand landmarks from the camera, and with the model classifies it as a gesture
    To run the demo simply execute "project.py" -->
