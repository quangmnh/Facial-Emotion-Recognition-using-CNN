import cv2
capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    cv2.imshow('Emotion :3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()