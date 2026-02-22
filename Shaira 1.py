import cv2
import numpy as np
import mediapipe as mp

# ================= CONFIGURATION =================
brushThickness = 15
eraserThickness = 50

# Colors (BGR format)
colors = {
    "PINK": (255, 0, 255),
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "ERASER": (0, 0, 0)
}

drawColor = colors["PINK"]
activeTool = "PINK"

# ================= MEDIAPIPE SETUP =================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ================= CAMERA SETUP =================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ================= CANVAS =================
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0

print("Virtual Painter Running... Press Q to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    # ================= HAND LANDMARKS =================
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

    # ================= FINGER LOGIC =================
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # Index tip
        x2, y2 = lmList[12][1:]  # Middle tip

        fingers = []

        # Thumb (horizontal check)
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in [8, 12, 16, 20]:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # ================= TOOL SELECTION =================
        if y1 < 125:
            if 250 < x1 < 450:
                activeTool = "PINK"
                drawColor = colors["PINK"]
            elif 550 < x1 < 750:
                activeTool = "BLUE"
                drawColor = colors["BLUE"]
            elif 800 < x1 < 950:
                activeTool = "GREEN"
                drawColor = colors["GREEN"]
            elif 1050 < x1 < 1200:
                activeTool = "ERASER"
                drawColor = colors["ERASER"]

            xp, yp = 0, 0

        # ================= DRAWING MODE =================
        elif fingers[1] and not fingers[2]:

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if activeTool == "ERASER" else brushThickness

            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)

            xp, yp = x1, y1

        else:
            xp, yp = 0, 0

    # ================= MERGE CANVAS =================
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # ================= HEADER UI =================
    cv2.rectangle(img, (0, 0), (1280, 125), (50, 50, 50), cv2.FILLED)

    def draw_button(x_min, x_max, color, tool_name):
        cv2.rectangle(img, (x_min, 10), (x_max, 115), color, cv2.FILLED)
        if activeTool == tool_name:
            cv2.rectangle(img, (x_min, 10), (x_max, 115), (255, 255, 255), 4)

    draw_button(250, 450, colors["PINK"], "PINK")
    draw_button(550, 750, colors["BLUE"], "BLUE")
    draw_button(800, 950, colors["GREEN"], "GREEN")
    draw_button(1050, 1200, (200, 200, 200), "ERASER")

    cv2.putText(img, "ERASER", (1065, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("AI Virtual Painter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()