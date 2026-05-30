- Real time had gesture detection: from rock paper scissors to sign interpretation

The project is comprised of 2 parts: the model and the tracker (the latter serves also as demo)

We remind you of the "requirements.txt" file with all the necessary programs and relative versions we used to run all the code 

    1. The model is trained from the merged_dataset.csv file that contains the hand landmarks of all the labeled gestures (rock, paper, scissors and thumbs up)
        To train the model simply run the "train_xgboost.py" file

    2. The tracker uses Mediapipe to recognize in real time the hand landmarks from the camera, and with the model classifies it as a gesture
        To run the demo simply execute "project.py"