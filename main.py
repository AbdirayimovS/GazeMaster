import cv2, gaze
import gaze_with_mean_of_both_pupil as mean_gaze
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh # TODO: face_mesh or mp_face_mesh



# Camera stream: 
cap = cv2.VideoCapture(0) # Camera index depens on your os, usually index 0 is ok
with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        _, image = cap.read()
        if not _:
            print("Empty camera frame :(")
            continue
        # To improve performance, optinally mark the image as not writeable to pass by reference
        image.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            # gaze.gaze(image, results.multi_face_landmarks[0]) # magic gaze estimation function
            mean_gaze.gaze(image, results.multi_face_landmarks[0]) # magic gaze estimation function
        
        flipped_image = cv2.flip(image, 1)
        cv2.imshow("Magic window", flipped_image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
        
cap.release()
cv2.destroyAllWindows()