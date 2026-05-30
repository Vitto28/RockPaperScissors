# Real time had gesture detection: from rock paper scissors to sign interpretation

This project's goal is building a computer vision system capable of recognizing a variety of **hand signs and gestures** captured by a static camera. It uses a robust gesture recognition pipeline that can distinguish between different hand configurations in real time. 

As a proof-of-concept, the system is integrated into a game of **rock-paper-scissors**, where it detects each player's gesture and determines the outcome of each round. However, the underlying technology can be extended to recognize a wide range of hand signs, making it applicable for various applications such as sign language interpretation, human-computer interaction, and more.

The project is comprised of 2 parts: the model pipeline and the real-time tracker integrated within the game (the latter serves also as demo)

## Installation and usage
You may want to use the `pip install -r requirements.txt` command to install all the packages you require with their versions.

> **Important**: for all the packages to work properly, you need to run the code with **Python 3.9**. We recommend using a virtual environment to manage the dependencies and avoid conflicts with other projects.

### Model pipeline
You can find the pre-trained model and the label encoder in the "model" folder (`rps_model.pkl` and `rps_label`, respectively). However, if you want to train the model on your own dataset, you may run `extract_landmarks.py` to extract the hand landmarks from **labeled images** of the gestures you want to recognize, and then run `train_xgboost.py` to train the model on the extracted landmarks.

> **Note**: you *may* run the game-integrated tracker with your own model, but it expects a set of specific gestures (rock, paper, scissors and thumbs up) to work properly. If you want to use it with a different set of gestures, you will need to modify the code accordingly.

Otherwise, here is the general workflow we followed to build the model and the tracker:
1. **Collecting the data**: we used a mixture of landmark and hand gesture datasets available online (the first from Zenodo, available here; the latter from Kaggle, available here). To extract the hand landmarks from the images, we used the `extract_landmarks.py` script, which uses Mediapipe to detect the hand landmarks and save them in a CSV file. We then merged the extracted landmarks with the original labels to create a final dataset (`merged_dataset.csv`) that contains both the landmarks and the corresponding gesture labels. We do this via the `merge_datasets.py` script.
2. **Data augmentation**: to increase the robustness of the model and prevent overfitting, we applied data augmentation techniques to the extracted landmarks. This included adding random noise, applying random rotations, and scaling the landmarks. We also **dropped** the z-axis of the landmarks, as it was not necessary for our task and could introduce noise into the model.
3. **Feature engineering**: we performed feature engineering on the extracted landmarks to create new features that could help the model better distinguish between different gestures. This included calculating the distances between specific landmarks, such as the tips of the fingers and the wrist, and the distance between all fingers.
4. **Model training**: we trained an XGBoost classifier and a Random Forest classifier on the extracted landmarks and the engineered features. While both models performed well, we chose to use the XGBoost model for the real-time tracker due to its slightly better performance.
5. **Game logic and real-time tracking**: we integrated the trained model into a real-time tracker that uses Mediapipe to detect the hand landmarks from the camera feed. The tracker then uses the trained model to classify the detected landmarks into one of the predefined gestures (rock, paper, scissors, or thumbs up). The game logic is implemented to determine the winner of each round based on the detected gestures.


<!-- We remind you of the "requirements.txt" file with all the necessary programs and relative versions we used to run all the code  -->

<!-- 1. The model is trained from the merged_dataset.csv file that contains the hand landmarks of all the labeled gestures (rock, paper, scissors and thumbs up)
    To train the model simply run the "train_xgboost.py" file

2. The tracker uses Mediapipe to recognize in real time the hand landmarks from the camera, and with the model classifies it as a gesture
    To run the demo simply execute "project.py" -->