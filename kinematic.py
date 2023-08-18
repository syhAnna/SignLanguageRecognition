from __future__ import division
import glob
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from util_hand import hand_to_trans_dict, rotate_y, rotate_x, rotate_z

results = None


def clean_data_to_df(direct):
    # print(direct)
    global results
    features = ['LeftHand',
                'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
                'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
                'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
                'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
                'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
                'IndexMiddle', 'ThumbNeighbor',
                ]
    # mp_drawing = mp.solutions.drawing_utils
    letter_files = glob.glob(direct + "*.jpg")
    mp_hands = mp.solutions.hands
    
    # For static images:
    image_files = letter_files
    landmark_results = []
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in tqdm(enumerate(image_files)):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmark_results.append(results)
    kinematic_rotation_table = []
    for i in range(len(letter_files)):
        lmk = landmark_results[i]
        if lmk.multi_handedness:
            trans_dict = hand_to_trans_dict(lmk.multi_handedness[0].classification[0].label,
                                            lmk.multi_hand_landmarks[0])
            kinematic_rotation_table.append([trans_dict[f] for f in features] + [letter_files[i].split("/")[-1][0]])
            # print(letter_files[i].split("/")[-1][0])
            # Calculate the position of the thumb tip

    df = pd.DataFrame(kinematic_rotation_table, columns=features + ["Label"])
    # print(df["LeftHandIndex2"][0])
    
    knuckle1s = ["LeftHandIndex1", "LeftHandMiddle1", "LeftHandRing1", "LeftHandPinky1", "LeftHandThumb1"]
    knuckle2s = ["LeftHandIndex2", "LeftHandMiddle2", "LeftHandRing2", "LeftHandPinky2", "LeftHandThumb2"]
    knuckle3s = ["LeftHandIndex3", "LeftHandMiddle3", "LeftHandRing3", "LeftHandPinky3", "LeftHandThumb3"]
    # Calculate the Y rotation
    for kn in knuckle1s:
        df[kn + "_rx"] = [rotate_x(x) for x in df[kn]]
        df[kn + "_ry"] = [rotate_y(x) for x in df[kn]]
        df[kn + "_rz"] = [rotate_z(x) for x in df[kn]]
    for kn in knuckle2s:
        df[kn + "_rx"] = [x[0] for x in df[kn]]
        df[kn + "_ry"] = [x[0] for x in df[kn]]
        df[kn + "_rz"] = [x[0] for x in df[kn]]
    for kn in knuckle3s:
        df[kn + "_rx"] = [x[0] for x in df[kn]]
        df[kn + "_ry"] = [x[0] for x in df[kn]]
        df[kn + "_rz"] = [x[0] for x in df[kn]]
    for f in features:
        df[f] = [x[0] for x in df[f]]

    new_columns = ['LeftHand',
                   'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
                   'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
                   'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
                   'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
                   'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
                   'LeftHandIndex1_rx', 'LeftHandMiddle1_rx', 'LeftHandRing1_rx', 'LeftHandPinky1_rx',
                   'LeftHandThumb1_rx', 'LeftHandIndex2_rx', 'LeftHandMiddle2_rx', 'LeftHandRing2_rx', 'LeftHandPinky2_rx',
                   'LeftHandThumb2_rx', 'LeftHandIndex3_rx', 'LeftHandMiddle3_rx', 'LeftHandRing3_rx', 'LeftHandPinky3_rx',
                   'LeftHandThumb3_rx',
                   'LeftHandIndex1_ry', 'LeftHandMiddle1_ry', 'LeftHandRing1_ry', 'LeftHandPinky1_ry',
                   'LeftHandThumb1_ry', 'LeftHandIndex2_ry', 'LeftHandMiddle2_ry', 'LeftHandRing2_ry', 'LeftHandPinky2_ry',
                   'LeftHandThumb2_ry', 'LeftHandIndex3_ry', 'LeftHandMiddle3_ry', 'LeftHandRing3_ry', 'LeftHandPinky3_ry',
                   'LeftHandThumb3_ry', 'LeftHandIndex1_rz', 'LeftHandMiddle1_rz', 'LeftHandRing1_rz', 'LeftHandPinky1_rz',
                   'LeftHandThumb1_rz', 'LeftHandIndex2_rz', 'LeftHandMiddle2_rz', 'LeftHandRing2_rz', 'LeftHandPinky2_rz',
                   'LeftHandThumb2_rz', 'LeftHandIndex3_rz', 'LeftHandMiddle3_rz', 'LeftHandRing3_rz', 'LeftHandPinky3_rz',
                   'LeftHandThumb3_rz', 'IndexMiddle', 'ThumbNeighbor',
                   "Label"]
    
    df = df[new_columns]
    
    return df

# train result
# data_dir = '/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/archive/asl_alphabet_train/asl_alphabet_train/{}/'  # Update the directory if you placed the dataset in different folder.
# big_letters = list(map(chr, range(ord('A'), ord('Z') + 1)))
# dfs = []
# for l in big_letters:
#     dfs.append(clean_data_to_df(data_dir.format(l)))
    
# df_result = dfs[0]
# for i in range(1, 26):
#     df_result = df_result.append(dfs[i])

# df_result = df_result.iloc[np.random.permutation(len(df_result))]

# print(df_result)

# df_result.to_pickle("/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/ik_annotated/kaggle_asl.pkl")


# test result
data_dir = '/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/archive/asl_alphabet_test/asl_alphabet_test/'  # Update the directory if you placed the dataset in different folder.

dfs = []
for i in range(26):
    dfs.append(clean_data_to_df(data_dir))
    
df_result = dfs[0]
for i in range(1, 26):
    df_result = df_result.append(dfs[i])

df_result = df_result.iloc[np.random.permutation(len(df_result))]

print(df_result)

df_result.to_pickle("/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/ik_annotated/kaggle_asl_test.pkl")