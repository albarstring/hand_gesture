import cv2
import mediapipe as mp

# Inisialisasi mediapipe hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Fungsi untuk menghitung jari yang terangkat
def count_fingers(hand_landmarks, image):
    h, w, _ = image.shape
    lm = hand_landmarks.landmark

    # Simpan hasil dalam bentuk list True/False untuk tiap jari
    finger_status = []

    # --- Jempol ---
    # Jempol agak horizontal, jadi bandingkan x, bukan y
    if lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x:
        finger_status.append(1)
    else:
        finger_status.append(0)

    # --- Empat jari lainnya (telunjuk - kelingking) ---
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_dips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    for tip, pip in zip(finger_tips, finger_dips):
        if lm[tip].y < lm[pip].y:  # ujung jari di atas ruas tengah
            finger_status.append(1)
        else:
            finger_status.append(0)

    # Hitung total jari terangkat
    total_fingers = sum(finger_status)
    return total_fingers

# Fungsi utama deteksi gesture
def detect_hand_number(image, hand):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(hand_landmarks, image)
            cv2.putText(image, f"Jari: {fingers_up}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    return image

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Tidak dapat membuka kamera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal menangkap frame")
        break

    frame = cv2.flip(frame, 1)
    frame = detect_hand_number(frame, hands)
    cv2.imshow("Deteksi Jumlah Jari (1-5)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
