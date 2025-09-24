import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

# Load doodle classification model
model = load_model("model.h5")
class_names = ['apple', 'book', 'cat', 'cup', 'triangle', 'tree', 'sun']

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing canvas
canvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = None, None
draw_color = (255, 0, 255)
thickness = 5
eraser_thickness = 300
label_image = None

# Load header
header = cv2.imread('updated_header_with_colors.png')
if header is None:
    print("❌ Header image not found. Using blank header.")
    header = np.zeros((100, 1280, 3), dtype=np.uint8)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    img[0:100, 0:1280] = header

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    if len(lmList) != 0:
        x1, y1 = lmList[8]
        x2, y2 = lmList[12]

        fingers_up = []
        if lmList[4][0] < lmList[3][0]:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        for tip_id in [8, 12, 16, 20]:
            if lmList[tip_id][1] < lmList[tip_id - 2][1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        if fingers_up[1] and fingers_up[2]:
            xp, yp = None, None
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv2.FILLED)

            if 0 < y1 < 100:
                if 0 < x1 < 256:
                    draw_color = (255, 0, 255)
                elif 256 < x1 < 512:
                    draw_color = (255, 255, 0)
                elif 512 < x1 < 768:
                    draw_color = (255, 0, 0)
                elif 768 < x1 < 1024:
                    draw_color = (0, 0, 0)
                elif 1024 < x1 < 1280:
                    draw_color = (255, 255, 255)

        elif fingers_up[1] and not fingers_up[2]:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            if xp is None and yp is None:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):  # eraser
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)

            xp, yp = x1, y1
    else:
        xp, yp = None, None

    # Combine camera and canvas
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Draw color selection box
    cv2.rectangle(img, (10, 10), (110, 90), draw_color, cv2.FILLED)

    # ==== Prediction and Clear key handling ====
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            roi = gray[y:y+h, x:x+w]

            if roi.size > 0:
                resized = cv2.resize(roi, (28, 28))
                resized = cv2.bitwise_not(resized)
                input_img = resized.reshape(1, 28, 28, 1).astype("float32") / 255.0

                pred = model.predict(input_img)[0]
                label = class_names[np.argmax(pred)]
                conf = np.max(pred) * 100

                cv2.putText(img, f"{label} ({conf:.1f}%)", (50, 680),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                print(f"Prediction: {label} ({conf:.1f}%)")

                # Load image by label
                label_image = None
                if label == 'cat':
                    label_image = cv2.imread('cat_image.png')
                elif label == 'triangle':
                    label_image = cv2.imread('triangle_image.png')
                elif label == 'cup':
                    label_image = cv2.imread('cup_image.png')
                elif label == 'book':
                    label_image = cv2.imread('book_image.png')
                elif label == 'apple':
                    label_image = cv2.imread('apple_image.png')
                elif label == 'tree':
                    label_image = cv2.imread('tree_image.png')
                elif label == 'sun':
                    label_image = cv2.imread('sun_image.png')

                # Put label image on canvas
                if label_image is not None:
                    label_resized = cv2.resize(label_image, (w, h))
                    y_end = min(y + h, canvas.shape[0])
                    x_end = min(x + w, canvas.shape[1])
                    img[y:y_end, x:x_end] = label_resized[0:(y_end - y), 0:(x_end - x)]

    elif key == ord('c'):
        canvas[:] = 0
        print("Canvas cleared")

    elif key == ord('q'):
        break

    cv2.imshow("Virtual Painter", img)

cap.release()
cv2.destroyAllWindows()
