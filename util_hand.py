import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion

# pip install pyquaternion
LEG_LENGTH = 0


def normalize(v):
    return v / np.linalg.norm(v)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def to_angle_axis(rotation_m):
    """
    Transform the rotation matrix to degrees and axis
    :param rotation_m: rotation matrix
    :return: [angle, x, y, z]
    """
    q = Quaternion(matrix=rotation_m.transpose())
    axis = q.axis
    return [q.degrees, axis[0], axis[1], axis[2]]


def middle(v1, v2):
    return (v1 + v2) / 2.


def vector_to_angle_axis(v):
    """
    Calculate the rotation according to the x and z axis transform y to v
    :param v: target vector
    :return: [angle, x, y, z]
    """
    y = np.asarray([0., 1., 0.])
    angle = angle_between(v, y)
    axis = np.cross(y, v)
    return [angle, axis[0], axis[1], axis[2]]


def angle_axis_to_matrix(vec4):
    rotation = Quaternion(axis=[vec4[1], vec4[2], vec4[3]], degrees=vec4[0])
    return rotation.rotation_matrix

def rotate_x(vec4):
    print("x is", vec4[0], "w is", vec4[3])
    return np.degrees(np.arctan(vec4[0] / vec4[3]))

def rotate_y(vec4):
    print("y is", vec4[1], "w is", vec4[3])
    return np.degrees(np.arctan(vec4[1] / vec4[3]))

def rotate_z(vec4):
    print("z is", vec4[2], "w is", vec4[3])
    return np.degrees(np.arctan(vec4[2] / vec4[3]))

def to_trans_dict(right_hand_landmarks=None):
    """
    An IK module to transform the pose landmarks to kinematic transformations
    :param from_server: Different landmark format from the server and mobile devices
    :param right_hand_landmarks: 21 hand landmarks from Mediapipe
    :return: the transformation dictionary used in the AR application
    """
    trans_dict = {}  
    # Right hand
    # right_hand_visibility = np.asarray([lmk.visibility for lmk in right_hand_landmarks.landmark])
    right_hand_landmark_list = np.asarray([[lmk.x, -lmk.y, -lmk.z] for lmk in right_hand_landmarks.landmark])
    hand_middle = (right_hand_landmark_list[5] + right_hand_landmark_list[17]) / 2.
    y = normalize(hand_middle - right_hand_landmark_list[0])
    z = normalize(np.cross(right_hand_landmark_list[5] - right_hand_landmark_list[0],
                            right_hand_landmark_list[17] - right_hand_landmark_list[0]))
    x = np.cross(y, z)
    right_hand_cs = np.asarray([x, y, z])
    # right_hand_rotation = np.matmul(right_hand_cs, inv(right_fore_arm_cs))
    # trans_dict["RightHand"] = to_angle_axis(right_hand_rotation)

    # Right hand fingers
    # Index1
    y = normalize(right_hand_landmark_list[6] - right_hand_landmark_list[5])
    y_p = np.matmul(y, inv(right_hand_cs))
    trans_dict["RightHandIndex1"] = vector_to_angle_axis(y_p)
    # Index2. Since the knuckles only rotate according to the x axis, we use the angle between function.
    angle = angle_between(right_hand_landmark_list[7] - right_hand_landmark_list[6],
                            right_hand_landmark_list[6] - right_hand_landmark_list[5])
    trans_dict["RightHandIndex2"] = [angle, 1.0, 0.0, 0.0]
    # Index3
    angle = angle_between(right_hand_landmark_list[8] - right_hand_landmark_list[7],
                            right_hand_landmark_list[7] - right_hand_landmark_list[6])
    trans_dict["RightHandIndex3"] = [angle, 1.0, 0.0, 0.0]

    # Middle1
    y = normalize(right_hand_landmark_list[10] - right_hand_landmark_list[9])
    y_p = np.matmul(y, inv(right_hand_cs))
    trans_dict["RightHandMiddle1"] = vector_to_angle_axis(y_p)
    # Middle2
    angle = angle_between(right_hand_landmark_list[11] - right_hand_landmark_list[10],
                            right_hand_landmark_list[10] - right_hand_landmark_list[9])
    trans_dict["RightHandMiddle2"] = [angle, 1.0, 0.0, 0.0]
    # Middle3
    angle = angle_between(right_hand_landmark_list[12] - right_hand_landmark_list[11],
                            right_hand_landmark_list[11] - right_hand_landmark_list[10])
    trans_dict["RightHandMiddle3"] = [angle, 1.0, 0.0, 0.0]

    # Ring1
    y = normalize(right_hand_landmark_list[14] - right_hand_landmark_list[13])
    y_p = np.matmul(y, inv(right_hand_cs))
    trans_dict["RightHandRing1"] = vector_to_angle_axis(y_p)
    # Ring2
    angle = angle_between(right_hand_landmark_list[15] - right_hand_landmark_list[14],
                            right_hand_landmark_list[14] - right_hand_landmark_list[13])
    trans_dict["RightHandRing2"] = [angle, 1.0, 0.0, 0.0]
    # Ring3
    angle = angle_between(right_hand_landmark_list[16] - right_hand_landmark_list[15],
                            right_hand_landmark_list[15] - right_hand_landmark_list[14])
    trans_dict["RightHandRing3"] = [angle, 1.0, 0.0, 0.0]

    # Pinky1
    y = normalize(right_hand_landmark_list[18] - right_hand_landmark_list[17])
    y_p = np.matmul(y, inv(right_hand_cs))
    trans_dict["RightHandPinky1"] = vector_to_angle_axis(y_p)
    # Pinky2
    angle = angle_between(right_hand_landmark_list[19] - right_hand_landmark_list[18],
                            right_hand_landmark_list[18] - right_hand_landmark_list[17])
    trans_dict["RightHandPinky2"] = [angle, 1.0, 0.0, 0.0]
    # Pinky3
    angle = angle_between(right_hand_landmark_list[20] - right_hand_landmark_list[19],
                            right_hand_landmark_list[19] - right_hand_landmark_list[18])
    trans_dict["RightHandPinky3"] = [angle, 1.0, 0.0, 0.0]

    # Thumb1
    y = normalize(right_hand_landmark_list[2] - right_hand_landmark_list[1])
    y_p = np.matmul(y, inv(right_hand_cs))
    trans_dict["RightHandThumb1"] = vector_to_angle_axis(y_p)
    angle_axis = trans_dict["RightHandThumb1"]
    r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
    right_thumb1_cs = np.matmul(r.rotation_matrix.transpose(), right_hand_cs)
    # Thumb2
    angle = angle_between(right_hand_landmark_list[3] - right_hand_landmark_list[2],
                            right_hand_landmark_list[2] - right_hand_landmark_list[1])
    y = normalize(right_hand_landmark_list[3] - right_hand_landmark_list[2])
    y_p = np.matmul(y, inv(right_thumb1_cs))
    if y_p[0] < 0:
        trans_dict["RightHandThumb2"] = [angle, 0.0, 0.0, 1.0]
    else:
        trans_dict["RightHandThumb2"] = [-angle, 0.0, 0.0, 1.0]
    angle_axis = trans_dict["RightHandThumb2"]
    r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
    right_thumb2_cs = np.matmul(r.rotation_matrix.transpose(), right_thumb1_cs)
    # Thumb3
    angle = angle_between(right_hand_landmark_list[4] - right_hand_landmark_list[3],
                            right_hand_landmark_list[3] - right_hand_landmark_list[2])
    y = normalize(right_hand_landmark_list[4] - right_hand_landmark_list[3])
    y_p = np.matmul(y, inv(right_thumb2_cs))
    if y_p[0] < 0:
        trans_dict["RightHandThumb3"] = [angle, 0.0, 0.0, 1.0]
    else:
        trans_dict["RightHandThumb3"] = [-angle, 0.0, 0.0, 1.0]
    return trans_dict


def closest_node(node, nodes):
    n = node.copy()
    nodes = np.asarray(nodes)
    nodes[4] = [10000, 100000, 10000]
    dist_2 = np.sum((nodes - n) ** 2, axis=1)
    return np.argmin(dist_2)


def hand_to_trans_dict(left_or_right, hand_landmarks):
    # If the hand landmarks are given, we optimize the gestures with the hand landmarks
    # Left hand
    trans_dict = {}
    left_hand_landmarks = hand_landmarks
    if left_or_right == "Left":
        # left_hand_visibility = np.asarray([l.visibility for l in left_hand_landmarks.landmark])
        left_hand_landmark_list = np.asarray([[lmk.x, -lmk.y, -lmk.z] for lmk in left_hand_landmarks.landmark])
        # Right hand
    else:
        # left_hand_visibility = np.asarray([l.visibility for l in left_hand_landmarks.landmark])
        left_hand_landmark_list = np.asarray([[lmk.x, lmk.y, -lmk.z] for lmk in left_hand_landmarks.landmark])
    left_fore_arm_cs = np.identity(3)
    hand_middle = (left_hand_landmark_list[5] + left_hand_landmark_list[17]) / 2.
    y = normalize(hand_middle - left_hand_landmark_list[0])
    z = normalize(np.cross(left_hand_landmark_list[17] - left_hand_landmark_list[0],
                           left_hand_landmark_list[5] - left_hand_landmark_list[0]))
    x = np.cross(y, z)
    left_hand_cs = np.asarray([x, y, z])
    left_hand_rotation = np.matmul(left_hand_cs, inv(left_fore_arm_cs))
    trans_dict["LeftHand"] = to_angle_axis(left_hand_rotation)

    # Left hand fingers
    # Index1
    y = normalize(left_hand_landmark_list[6] - left_hand_landmark_list[5])
    y_p = np.matmul(y, inv(left_hand_cs))
    trans_dict["LeftHandIndex1"] = vector_to_angle_axis(y_p)
    # Index2. Since the knuckles only rotate according to the x axis, we use the angle between function.
    angle = angle_between(left_hand_landmark_list[7] - left_hand_landmark_list[6],
                          left_hand_landmark_list[6] - left_hand_landmark_list[5])
    trans_dict["LeftHandIndex2"] = [angle, 1.0, 0.0, 0.0]
    # Index3
    angle = angle_between(left_hand_landmark_list[8] - left_hand_landmark_list[7],
                          left_hand_landmark_list[7] - left_hand_landmark_list[6])
    trans_dict["LeftHandIndex3"] = [angle, 1.0, 0.0, 0.0]

    # Middle1
    y = normalize(left_hand_landmark_list[10] - left_hand_landmark_list[9])
    y_p = np.matmul(y, inv(left_hand_cs))
    trans_dict["LeftHandMiddle1"] = vector_to_angle_axis(y_p)
    # Middle2
    angle = angle_between(left_hand_landmark_list[11] - left_hand_landmark_list[10],
                          left_hand_landmark_list[10] - left_hand_landmark_list[9])
    trans_dict["LeftHandMiddle2"] = [angle, 1.0, 0.0, 0.0]
    # Middle3
    angle = angle_between(left_hand_landmark_list[12] - left_hand_landmark_list[11],
                          left_hand_landmark_list[11] - left_hand_landmark_list[10])
    trans_dict["LeftHandMiddle3"] = [angle, 1.0, 0.0, 0.0]

    # Ring1
    y = normalize(left_hand_landmark_list[14] - left_hand_landmark_list[13])
    y_p = np.matmul(y, inv(left_hand_cs))
    trans_dict["LeftHandRing1"] = vector_to_angle_axis(y_p)
    # Ring2
    angle = angle_between(left_hand_landmark_list[15] - left_hand_landmark_list[14],
                          left_hand_landmark_list[14] - left_hand_landmark_list[13])
    trans_dict["LeftHandRing2"] = [angle, 1.0, 0.0, 0.0]
    # Ring3
    angle = angle_between(left_hand_landmark_list[16] - left_hand_landmark_list[15],
                          left_hand_landmark_list[15] - left_hand_landmark_list[14])
    trans_dict["LeftHandRing3"] = [angle, 1.0, 0.0, 0.0]

    # Pinky1
    y = normalize(left_hand_landmark_list[18] - left_hand_landmark_list[17])
    y_p = np.matmul(y, inv(left_hand_cs))
    trans_dict["LeftHandPinky1"] = vector_to_angle_axis(y_p)
    # Pinky2
    angle = angle_between(left_hand_landmark_list[19] - left_hand_landmark_list[18],
                          left_hand_landmark_list[18] - left_hand_landmark_list[17])
    trans_dict["LeftHandPinky2"] = [angle, 1.0, 0.0, 0.0]
    # Pinky3
    angle = angle_between(left_hand_landmark_list[20] - left_hand_landmark_list[19],
                          left_hand_landmark_list[19] - left_hand_landmark_list[18])
    trans_dict["LeftHandPinky3"] = [angle, 1.0, 0.0, 0.0]

    # Thumb1
    y = normalize(left_hand_landmark_list[2] - left_hand_landmark_list[1])
    y_p = np.matmul(y, inv(left_hand_cs))
    trans_dict["LeftHandThumb1"] = vector_to_angle_axis(y_p)
    angle_axis = trans_dict["LeftHandThumb1"]
    r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
    left_thumb1_cs = np.matmul(r.rotation_matrix.transpose(), left_hand_cs)
    # Thumb2
    angle = angle_between(left_hand_landmark_list[3] - left_hand_landmark_list[2],
                          left_hand_landmark_list[2] - left_hand_landmark_list[1])
    y = normalize(left_hand_landmark_list[3] - left_hand_landmark_list[2])
    y_p = np.matmul(y, inv(left_thumb1_cs))
    if y_p[0] < 0:
        trans_dict["LeftHandThumb2"] = [angle, 0.0, 0.0, 1.0]
    else:
        trans_dict["LeftHandThumb2"] = [-angle, 0.0, 0.0, 1.0]
    angle_axis = trans_dict["LeftHandThumb2"]
    r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
    left_thumb2_cs = np.matmul(r.rotation_matrix.transpose(), left_thumb1_cs)
    # Thumb3
    angle = angle_between(left_hand_landmark_list[4] - left_hand_landmark_list[3],
                          left_hand_landmark_list[3] - left_hand_landmark_list[2])
    y = normalize(left_hand_landmark_list[4] - left_hand_landmark_list[3])
    y_p = np.matmul(y, inv(left_thumb2_cs))
    if y_p[0] < 0:
        trans_dict["LeftHandThumb3"] = [angle, 0.0, 0.0, 1.0]
    else:
        trans_dict["LeftHandThumb3"] = [-angle, 0.0, 0.0, 1.0]

    # Angle between the index and middle
    trans_dict["IndexMiddle"] = [angle_between(left_hand_landmark_list[8] - left_hand_landmark_list[5],
                                               left_hand_landmark_list[12] - left_hand_landmark_list[9]), 0, 0, 0]

    # The joint that closest to the thumb tip
    trans_dict["ThumbNeighbor"] = [closest_node(left_hand_landmark_list[4], left_hand_landmark_list), 0, 0, 0]
    return trans_dict


def angle_axis_to_string(angle_axis):
    return "{:.3f},{:.5f},{:.5f},{:.5f}".format(*angle_axis)