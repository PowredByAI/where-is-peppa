import cv2
import numpy as np
import imutils
import paddlex as pdx
model = pdx.deploy.Predictor('./peppa_model', use_gpu=False)
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
COLORS = np.random.uniform(0, 255, size=(10, 3))

while(cap.isOpened()):

    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    # Display the resulting frame
    (h, w) = frame.shape[:2]
    im = frame.astype('float32')
    result = model.predict(im)

    if model.model_type == "detector":
        for item in result:
            if item['score'] > 0.07:
                print(item)
                score = item['score']
                category = item['category']
                label = "{}: {:.2f}%".format(category, score * 100)
                bbox = item['bbox']
                startX = int(bbox[0])
                startY = int(bbox[1])
                width = int(bbox[2])
                height = int(bbox[3])
                endX = startX+width
                endY = startY+height

                print("{}-{}-{}-{}".format(startX, startY, endX, endY))
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 0, 255), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('', frame)
    # the 'q' button is set as the
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
