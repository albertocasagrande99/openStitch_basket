import cv2
video = cv2.VideoCapture('stitch.mp4')
cv2.namedWindow('Video')

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

cv2.setMouseCallback('Video', mouse_callback)

while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        break
