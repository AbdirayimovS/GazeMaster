import cv2
import numpy as np 
from helpers import relative, relativeT

def gaze(frame, points):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    
    2D image points.
    relative() takes mediapipe points that is normalized to [-1, 1] and
      returns image points at (x, y) format"""
    
    image_points = np.array([
        relative(points.landmark[4], frame.shape), # Nose tip
        relative(points.landmark[152], frame.shape), # Chin
        relative(points.landmark[263], frame.shape), # Lefte eye left corner
        relative(points.landmark[33], frame.shape), # Right eye right corner
        relative(points.landmark[287], frame.shape), # Left Mouth Corner
        relative(points.landmark[57], frame.shape) #  Right Mouth Corner
    ], dtype='double')

    """
    2D image points.
    relativeT() takes mediapipe points that is normalized to [-1, 1] and 
    returns image points at (x, y, 0) format
    """
    #TODO: What is main goal of relativeT function? Explain in one sentence
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape), # Nose tip
        relativeT(points.landmark[152], frame.shape), # Chin
        relativeT(points.landmark[263], frame.shape), # left eye left corner
        relativeT(points.landmark[33], frame.shape), # right eye right corner
        relativeT(points.landmark[287], frame.shape), # left mouth corner
        relativeT(points.landmark[57], frame.shape) # right mouth corner
    ], dtype='double')

    """
    3D model points. 
    Constants that are predefined from somewhere. 
    TODO: Why we need these 3D points how he defined them?
    The head size of adult is DIFFERENT from head size of CHILD 18-36 month old!
    """
    model_points = np.array([
        (0.0, 0.0, 0.0), # Nose tip
        (0, -63.6, -12.5), # Chin 
        (-43.3, 32.7, -26), # Left eye left corner
        (43.3, 32.7, -26), #Right eye, right corner
        (-28.9, - 28.9, -24.1), # Left Mouth corner
        (28.9, -28.9, -24.1) # right eye right corner
    ])

    """
    3D model eye points
    The center of the eye ball
    """
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]]) # the ceneter of the right eyeball as a vector
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]]) # the center of the left eyeball as a vector

    """
    Camera matric estimation. 
    TODO: Does it correlate with the height of CHILD? What is it in simple terms?
    """
    focal_length = frame.shape[1] # It is the width size of the frame. Why it is needed? 
    center  = (frame.shape[1] / 2, frame.shape[0] / 2)

    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double'
    )

    dist_coeffs = np.zeros((4, 1)) #TODO: here we assumed no lens disortion. What is actual lens disorder in our system?
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points,
                                                                  camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    
    # 2D pupil location
    left_pupil = relative(points.landmark[468], frame.shape) # Location of left pupil from mediapipe landmarks representatives
    right_pupil = relative(points.landmark[473], frame.shape) # Location of the right pupil respectively
    print("DEBUG: location of left pupil: ", left_pupil, "location of right pupil: ", right_pupil)
    # Transformation between image points to world points
    _, transformation, _  = cv2.estimateAffine3D(image_points1, model_points) #TODO: What is estimateAffine3D? Explain it briefly

    if transformation is not None: # if transformation is not None. There is also 0. Anyway it must return 0 too 
        pupil_world_cord_left = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T 
        pupil_world_cord_right = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T 


        # 3D gaze points (10 is arbitrary value denoting gaze distance)
        # I have used ruler and Iris dataset in JS on web app to measure distance between camera and person (depth estimation)
        S_L = Eye_ball_center_left + (pupil_world_cord_left - Eye_ball_center_left) * 60
        S_R = Eye_ball_center_right + (pupil_world_cord_right - Eye_ball_center_right) * 60


        # Project 3D gaze direction onto the image plane.
        (eye_pupil2D_L, _) = cv2.projectPoints((int(S_L[0]), int(S_L[1]), int(S_L[2])),
                                             rotation_vector,
                                             translation_vector,
                                             camera_matrix,
                                             dist_coeffs)
        
        # Project 3D gaze direction onto the image plane.
        (eye_pupil2D_R, _) = cv2.projectPoints((int(S_R[0]), int(S_R[1]), int(S_R[2])),
                                             rotation_vector,
                                             translation_vector,
                                             camera_matrix,
                                             dist_coeffs)

        # Project 3D head pose into the image plane 
        (head_pose_L, _) = cv2.projectPoints((int(pupil_world_cord_left[0]), int(pupil_world_cord_left[1]), 40),
                                            rotation_vector,
                                            translation_vector,
                                            camera_matrix,
                                            dist_coeffs)
        (head_pose_R, _) = cv2.projectPoints((int(pupil_world_cord_left[0]), int(pupil_world_cord_left[1]), 40),
                                            rotation_vector,
                                            translation_vector,
                                            camera_matrix,
                                            dist_coeffs)
        print("DEBUG: head_pose_L: ", type(head_pose_L), "head_pose_R: ", head_pose_R)
        print("DEBUG: head_pose[0][0]: ", np.mean([head_pose_L[0][0], head_pose_R[0][0]], axis=0))
        print("DEBUG: #########################")
        head_pose = np.mean([head_pose_L[0][0], head_pose_R[0][0]], axis=0)
        # Correct gaze for head rotation
        gaze_L = left_pupil + (eye_pupil2D_L[0][0] - left_pupil) - (head_pose - left_pupil)
        gaze_R = right_pupil + (eye_pupil2D_R[0][0] - right_pupil) - (head_pose - right_pupil)
        mean_gaze = ((gaze_L[0] + gaze_R[0]) / 2, (gaze_R[1] + gaze_L[1]) / 2)
        # Draw gaze line into screen 
        left_pupil_loc = (int(left_pupil[0]), int(left_pupil[1]))
        right_pupil_loc = (int(right_pupil[0]), int(right_pupil[1]))
        p2 = (int(mean_gaze[0]), int(mean_gaze[1]))
        cv2.line(frame, left_pupil_loc, p2, (0, 0, 255), 2)
        cv2.line(frame, right_pupil_loc, p2, (0, 0, 255), 2)
