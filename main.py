import pigpio
import time
import cv2 as cv
import numpy as np
import threading

pi = pigpio.pi();

mouth_open = 34
mouth_closed = 150
servo1 = 12
servo2 = 13

#MOTION DETECTION
# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 2000

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 50




def setupPins():
    pi.write(servo1, 0);
    pi.write(servo2, 0);
    pi.set_PWM_frequency(servo1, 200);
    pi.set_PWM_frequency(servo2, 200);

def open_mouth():
    pi.set_PWM_dutycycle(servo1, mouth_open);
    pi.set_PWM_dutycycle(servo2, mouth_open);
    time.sleep(0.5)
    pi.write(servo1, 0);
    pi.write(servo2, 0);

def close_mouth():
    pi.set_PWM_dutycycle(servo1, mouth_closed);
    pi.set_PWM_dutycycle(servo2, mouth_closed);
    time.sleep(0.5)
    pi.write(servo1, 0);
    pi.write(servo2, 0);



setupPins();

cap = cv.VideoCapture(0)
kernel = np.ones((5, 5))

first_frame = None
next_frame = None

delay_counter = 0
movement_persistent_counter = 0

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    transient_movement_flag = False
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blur it to remove camera noise (reducing false positives)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is nothing, initialise it
    if first_frame is None: first_frame = gray    

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appriopriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

        
    # Set the next frame to compare (the current frame)
    next_frame = gray

    # Compare the two frames, find the difference
    frame_delta = cv.absdiff(first_frame, next_frame)
    thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]

    # Fill in holes via dilate(), and find contours of the thesholds
    thresh = cv.dilate(thresh, None, iterations = 2)
    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv.boundingRect(c)
        
        # If the contour is too small, ignore it, otherwise, there's transient
        # movement
        if cv.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            transient_movement_flag = True
            
            # Draw a rectangle around big enough movements
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # The moment something moves momentarily, reset the persistent
    # movement timer.
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
        x = threading.Thread(target=open_mouth)
        x.start();

    # As long as there was a recent transient movement, say a movement
    # was detected    
    if movement_persistent_counter > 0:
        print( "Movement Detected " + str(movement_persistent_counter) )
        
        movement_persistent_counter -= 1
        if movement_persistent_counter == 0 :
            x = threading.Thread(target=close_mouth)
            x.start()
            print("No Movement Detected")
        
    
    # For if you want to show the individual video frames
#    cv.imshow("frame", frame)
#    cv.imshow("delta", frame_delta)
    
    # Convert the frame_delta to color for splicing
    frame_delta = cv.cvtColor(frame_delta, cv.COLOR_GRAY2BGR)

    # Splice the two video frames together to make one long horizontal one
    cv.imshow("frame", np.hstack((frame_delta, frame)))

    
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

