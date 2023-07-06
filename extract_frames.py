# create a folder to store extracted images
import os
import cv2

folder = 'FramesRight'  
os.mkdir(folder)
# use opencv to do the job
vidcap = cv2.VideoCapture('Video/right.mp4')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}_right.jpg".format(count)), image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,folder))
