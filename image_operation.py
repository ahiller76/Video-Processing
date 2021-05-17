### Aaron Hiller
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initializing frame count
framecnt=0

# Defining camera parameters
capture = cv2.VideoCapture(0)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Define file names
video = cv2.VideoWriter('live_camera.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, size)
videofilter = cv2.VideoWriter('modified_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, size)

# Video recording
while framecnt <= 4500:
    has_frame, frame = capture.read()
    if has_frame:
        framecnt = framecnt+1
    if not has_frame:
        print('Can\'t get frame')
        break

    video.write(frame)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27:
        print('Pressed Esc')
        break

capture.release()
video.release()
cv2.destroyAllWindows()

# Define video capture
capture = cv2.VideoCapture('live_camera.avi')

# Initialize plot
plt.ion()
plt.show()

# Video playback
while True:
    has_frame, vidframe = capture.read()
    if not has_frame:
        print('Reached end of video')
        break

    # Displaying frame
    cv2.imshow('frame', vidframe)

    # Creating blue frame
    b = vidframe.copy()
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    # Creating green frame
    g = vidframe.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    # Creating red frame
    r = vidframe.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    # Displaying RGB frames
    cv2.imshow('RGB-R', r)
    cv2.imshow('RGB-G', g)
    cv2.imshow('RGB-B', b)

    # Print shape
    print('Shape:', vidframe.shape)

    # Image scaling
    scaled_frame = cv2.resize(vidframe, (int(frame_width*.7), int(frame_height*.7)))
    cv2.imshow("Scaled frame", scaled_frame)

    # RGB to HSV conversion
    hsv = cv2.cvtColor(vidframe, cv2.COLOR_BGR2HSV)
    cv2.imshow('RGB to HSV', hsv)

    # Histogram of S channel
    plt.figure(1)
    plt.clf()
    hist, bins = np.histogram(hsv[...,1], 256, [0, 255])
    plt.fill_between(range(256), hist)
    plt.title('Saturation Histogram')

    # Equalized histogram of S channel
    plt.figure(2)
    plt.clf()
    hsv_eq = hsv.copy()
    hsv_eq[...,1] = cv2.equalizeHist(hsv_eq[...,1])
    hist_eq, bins = np.histogram(hsv_eq[...,1], 256, [0, 255])
    plt.fill_between(range(256), hist_eq)
    plt.title('Equalized Saturation Histogram')
    plt.draw_all()

    # HSV to RGB conversion
    from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV to RGB', from_hsv)

    # Gamma correction
    gamma = np.power(vidframe , .5)
    cv2.imshow('Gamma Correction', gamma)

    # Gaussian filer
    gauss_blur = cv2.GaussianBlur(vidframe, (7, 7), 0)
    cv2.imshow('Gaussian filer', gauss_blur)
    videofilter.write(gauss_blur)

    key = cv2.waitKey(1)
    if key == 27:
        print('Pressed Esc')
        break
plt.show()
cv2.destroyAllWindows()
videofilter.release()