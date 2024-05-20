# -*- coding: utf-8 -*-
# Helper Library
import reference_world as world
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import time
# Face Recognition libraries
import mediapipe as mp
import dlib
import cv2
# GUI library, Map
from tkinter import *
import tkintermapview
from geopy.distance import geodesic
# Communication module
import RPi.GPIO as GPIO
import serial

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.IN)  #  GPIO26 - Pin 37 on Raspberry Pi 4
GPIO.setup(13, GPIO.OUT)
GPIO.output(13, GPIO.LOW)

# Function to calculate EAR and MAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def mouth_aspects_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[3], mouth[9])
    average = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = average / D
    return mar

# Function to draw and display information on screen
def calculate_avg_ear(ear_values):
    return np.mean(ear_values)
def eyes_close():
    cv2.putText(image, "Eye: {}".format("close"), (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "MAR: {:.2f}".format(mar), (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
def tired():
    cv2.putText(image, "Tired!!!", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    GPIO.output(13, GPIO.HIGH)
def cautions():
    cv2.putText(image, "Caution!!!", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    GPIO.output(13, GPIO.HIGH)
def drawing_info():
    cv2.putText(image, "Eye: {}".format("Open"), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "MAR: {:.2f}".format(mar), (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
def drawing_line_and_xyz():
    cv2.line(image, p1, p2, (255, 0, 0), 3)
    cv2.putText(image, GAZE, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
def no_faces():
    cv2.putText(image, "NO FACE!", (0, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    GPIO.output(13, GPIO.HIGH)

# Usage thresh hold and parameter
flag = 1        # Flag to convert between display MAP and display Face recognition window

EAR_HISTORY_SIZE = 30  # 30 ear parameter on a list
ear_history = []        # List variable for ear

EYE_AR_THRESH = 0.2  # Eye thresh hold when begin to program (adjustable)
TIRED_THRESHOLD = 1.2  # Threshold for tiredness (in seconds)
NO_FACE_THRESHOLD = 1  # Threshold for no FACE (in seconds)
MOUTH_AR_THRESHOLD = 0.7  # Threshold for YAWN (in seconds)

COUNTER_CLOSE = 0       # contain counter (in second) for closing eye
COUNTER_FACE = 0        # contain counter (in second) for no face detect

start_time1 = None      # start time counting for no face
start_time2 = None      # start time counting for closing eye

yawn = 0                # Yawn parameter
yawn_time = 5           # yawn couter
yawn_count = 3

UPDATE_INTERVAL_EAR = 10  # Updated EAR threshold every 60 second
last_update_time_ear = time.time()      # store the last time ear being updated

# Setup face recognition module
detector = dlib.get_frontal_face_detector()  # using HOG method to detect face (detect face)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # using module to predict 68 landmarks

mp_face_mesh = mp.solutions.face_mesh  # import module "face_mesh" and attach it to :  mp_face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)  # detect face using mediapipe

mp_drawing = mp.solutions.drawing_utils  # import module "drawing_utils" to connect landmark
drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)  # spect for landmarkline

# Setup special landmark points
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # store list of the number on left eye (42-47)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # store list of the number on right eye (36-41)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]      # store list of the number on right eye (48-59)

# Setting up camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 60)
current_window = None
#####################################----MAP------#####################################
# Initialize serial connection with Arduino
ser = serial.Serial("/dev/ttyACM0", 9600, timeout = 1.0)
time.sleep(3)
ser.reset_input_buffer()
print("Serial Ok")

# Latitude and longitude of location for simulate
rest_locations = [
    (10.851332294802038, 106.7705222908667),
    (10.850673608868291, 106.7731138720598),
    (10.849839335885783, 106.77172520604105),
    (10.849965541131064, 106.77000616934157)
]

# Store previous latitude and longitude values
prev_latitude = None
prev_longitude = None
latitude = 10.848426274753104
longitude = 106.75211706645679
a = 0

# Function to update MAP
def update_map():
    global prev_latitude, prev_longitude
    try:
        # Read data from the serial port
        data = ser.readline().decode('utf-8').rstrip()
        print(data)
        if data.startswith("+CGNSINF"):
            # Split the data to extract coordinates
            parts = data.split(',')
            if len(parts) >= 5:
                latitude = float(parts[3])  # Latitude
                longitude = float(parts[4])  # Longitude

                # Check if the difference between the new and previous values is less than 0.1 units
                if prev_latitude is not None and prev_longitude is not None:
                    if abs(latitude - prev_latitude) > 0.1:
                        latitude = prev_latitude
                    if abs(longitude - prev_longitude) > 0.1:
                        longitude = prev_longitude

                # Update previous latitude and longitude values
                prev_latitude = latitude
                prev_longitude = longitude
                map_widget.set_position(latitude, longitude)
                # Print out the coordinates
                print("Latitude:", latitude)
                print("Longitude:", longitude)
                current_location = [latitude, longitude]
                # Delete all markers
                map_widget.delete_all_marker()

                # Set the position of map and set marker on the location get from serial
                map_widget.set_marker(latitude, longitude, text="My Location")

                # Calculate nearest location
                nearest_location = None
                nearest_distance = float('inf')
                for i, location in enumerate(rest_locations):
                    distance = geodesic(current_location, location).meters
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_location = location

                # Set marker for 4 simulate locations
                for i, location in enumerate(rest_locations):
                    if location == nearest_location:
                        marker = map_widget.set_marker(location[0], location[1], text=f"Location {i + 1}-nearest",
                                                       font=("Arial", 12))
                    else:
                        marker = map_widget.set_marker(location[0], location[1], text=f"Location {i + 1}",
                                                       font=("Arial", 12))
    except ValueError:
        print("Invalid data format:", data)
    root.after(1000, update_map)  # Run update_map function after 3 seconds

# Function to convert to Face recognition window went button is press
def press_button():
    global flag
    global COUNTER_CLOSE
    global start_time2
    global image
    global root
    global a
    flag = 1
    COUNTER_CLOSE = 0
    start_time2 = None
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    a = 0
    root.after_cancel(update_map)
    root.quit()
    root = None

def mute():
    global a
    a = 1 - a

def check_condition():
    if a == 0: 
        GPIO.output(13, GPIO.HIGH)
    else:
        GPIO.output(13, GPIO.LOW)
    root.after(50, check_condition)
#####################################----MAIN LOOP----#####################################
while True:
    # Read the state of GPIO pin 26
    time.sleep(0.1)
    gpio_input26 = GPIO.input(26)
    if gpio_input26 == GPIO.HIGH:  # State is HIGH - have information from CAN system
        time.sleep(0.1)
        image = cap.read()
        if flag == 1:  # FLAG variable to convert between Face recognition vs MAP
            time.sleep(0.3)
            a = 0
            GPIO.output(13, GPIO.LOW)
            current_time = time.time()  # start counting time
            GAZE = ""
            face_3d = []                # list of 3d face point
            face_2d = []                # list of 2d face point
            success, image = cap.read()  # get the image from camera
            image = cv2.flip(image, 1)  # flip the image
            if not success:  # Error from camera
                print(f'[ERROR - System] KHONG THE KET NOI CAMERA')
                break

            image = imutils.resize(image, width=int(720))  # resize image
            image.flags.writeable = False  # set read-only to the image

            # Draw the face detection annotations on the image.
            image.flags.writeable = True  # set image to be adjustable
            results = face_mesh.process(image)  # get face detect result from mediapipe module
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert color from RGB to BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to gray scale

            img_h, img_w, img_c = image.shape  # get the height, width, channel of the image
            faces = detector(gray, 0)  # detect face in gray image

            if len(faces) > 0 and results.multi_face_landmarks:  # have detected face or results in not None
                # reset when have face
                COUNTER_FACE = 0
                start_time1 = None

                # loop through all faces list
                for face in faces:
                    shape = predictor(gray, face)   # predict the landmark
                    shape = face_utils.shape_to_np(shape)   # create a numpy array contain coordinate (x,y) of landmark point

                    # Get the coordinate of left eye, right eye and mouth
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    mouth = shape[mStart:mEnd]

                    # Calculate EAR and MAR
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    mar = mouth_aspects_ratio(mouth)
                    ear = (leftEAR + rightEAR) / 2.0
                    ear_history.append(ear)

                    # Drawing eyes and mouth
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    mouthHull = cv2.convexHull(mouth)

                    cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(image, [mouthHull], -1, (0, 0, 255), 1)

                    # Caculate EAR and updated EAR
                    if ear >= (EYE_AR_THRESH - 0.035):
                        if len(ear_history) < EAR_HISTORY_SIZE:
                            ear_history.append(ear)
                        else:
                            ear_history.pop(0)
                            ear_history.append(ear)
                            EYE_AR_THRESH = calculate_avg_ear(ear_history)

                    # Update EAR every 60 second
                    if current_time - last_update_time_ear >= UPDATE_INTERVAL_EAR:
                        EYE_AR_THRESH = calculate_avg_ear(ear_history)
                        last_update_time_ear = current_time

                    # Check if eyes are closing
                    if ear < (EYE_AR_THRESH - 0.035):
                        # Run eyes_close function
                        eyes_close()
                        if start_time2 is None:
                            start_time2 = time.time()
                        COUNTER_CLOSE = time.time() - start_time2
                        # Check if the eyes were closed for a prolonged period if toolong open MAP
                        if COUNTER_CLOSE > TIRED_THRESHOLD:
                            tired()
                            flag = 0  # Display MAP
                    # Check if mouth is Yawning for a prolonged period if toolong open MAP
                    elif mar > MOUTH_AR_THRESHOLD:
                        yawn += 1
                        if yawn > yawn_time:
                            # Run cautions function
                            cautions()
                            yawn_count += 1
                            # If Yawn 3 time
                            if yawn_count >= 3:
                                flag = 0  # Display MAP

                    # Driver in normal condition
                    else:
                        # Rung drawing_info function and reset all parameter
                        drawing_info()
                        COUNTER_CLOSE = 0
                        start_time2 = None

                # Loop through all the face
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):  # Loop through all index and coordinate of landmark lap qua index va toa do (***) cua cac landmark
                        # lm.x lm.y value is left right coordinate, the middle line of the screen is 0.5
                        # lm.x to the left the value is decrease but still > 0, to the right the value is increase but < 1
                        # similar lm.y value will decrease when go up, lm.y value will increase when go down
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:  # filter special landmark points
                        #  chin  left eye corner  nose tip  right eye corner  left mouth corner  right mouth corner
                            if idx == 1:
                                # convert left right coordinate to Oxy coordinate
                                nose_2d = (lm.x * img_w,lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)  # Let x,y contain Oxy coordinate of 6 special points

                            # Get the 2D Coordinates
                            face_2d.append([x, y])  # append to list face_2d[]
                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])  # append to list face_3d[]

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)
                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # Camera parameter
                    focal_length = img_w * 1
                    cameraMatrix = world.cameraMatrix(focal_length, img_h / 2,img_w / 2)
                    # Store Lens deviation parameters
                    mdists = np.zeros((4, 1), dtype=np.float64)

                    # Calculate rotation and translation vector using solvePnP
                    success, rotationVector, translationVector = cv2.solvePnP(
                        face_3d, face_2d, cameraMatrix, mdists)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rotationVector)

                    # Get angles thanks to rmat (rotation vector)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    # angles around Oxy coordinate axes

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    # See where the driver's head tilting
                    if y < -5 < x < 5 and -7:
                        GAZE = "Looking Left"
                    elif -5 < x < 5 and y > 7:
                        GAZE = "Looking Right"
                    elif x < -5:
                        GAZE = "Looking Down"
                    elif x > 5:
                        GAZE = "Looking Up"
                    elif -5 < x < 5 or -7 < y < 7:
                        GAZE = "Forward"

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotationVector, translationVector,
                                                                     cameraMatrix,
                                                                     mdists)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                    # Run drawing_line_and_xyz function
                    drawing_line_and_xyz()
            # No faces were detected
            elif len(faces) == 0 or results.multi_face_landmarks is None:
                # Start counting no face time
                if start_time1 is None:
                    start_time1 = time.time()
                COUNTER_FACE = time.time() - start_time1

                # If no faces were detected for a certain period of time
                if COUNTER_FACE > NO_FACE_THRESHOLD:
                    # Run no_faces function
                    no_faces()

            # Convert color from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the window
            # If current window not name "Khuon mat"
            if current_window != "Khuon mat":
                cv2.destroyAllWindows()  # close all window
                cv2.namedWindow("Khuon mat", cv2.WND_PROP_FULLSCREEN)   # Setting full screen
                cv2.setWindowProperty("Khuon mat", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Khuon mat", image) # Open window
                current_window = "Khuon mat"
            # If current window name "Khuon mat"
            else:
                cv2.namedWindow("Khuon mat", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Khuon mat", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Khuon mat", image)
            if flag == 0:
                image = None
                yawn = 0
                COUNTER_CLOSE = 0
                start_time2 = None
            else :
                image = image
        # If detect tired
        elif flag == 0:
            time.sleep(0.1)
            cv2.destroyAllWindows() # close all window
            
            # Setting up tkinter window
            root = Tk()
            root.title('Map')
            root.attributes('-fullscreen', True)

            # Frame
            my_frame = Frame(root)
            my_frame.pack(pady=40)
            map_widget = tkintermapview.TkinterMapView(my_frame, width=800, height=600)
            map_widget.set_zoom(16)
            check_condition()
            update_map()

            close_button = Button(my_frame, text='Close', command=press_button,  width=50, height=2)
            mute= Button(my_frame, text='Mute', command= mute, width=50, height=2)
            close_button.place(x=10, y=10)  
            mute.place(x=100, y=10)  
            mute.pack()
            close_button.pack()
            map_widget.pack()
            
            
            root.mainloop()

    # State from GPIO26 is Low - don't have information from CAN system
    else:
        time.sleep(0.1)
        GPIO.output(13, GPIO.LOW)
        if current_window != "Blue":
            cv2.destroyAllWindows()  #
            image = np.zeros((480, 640, 3), dtype=np.uint8)  #
            cv2.putText(image, "Xe Dang Dung Yen", (180, 235), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.namedWindow("Blue", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Blue", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Blue", image)
            current_window = "Blue"
        else:
            cv2.namedWindow("Blue", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Blue", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Blue", image)

    # Check if "q" button on keyboard were pressed close the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        GPIO.output(13, GPIO.LOW)
        break

GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
