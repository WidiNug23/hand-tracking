import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands

# Konfigurasi deteksi yang optimal
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indeks landmark untuk 5 ujung jari: [Jempol, Telunjuk, Tengah, Manis, Kelingking]
FINGER_TIPS = [4, 8, 12, 16, 20]

# Variabel untuk Smoothing & Interaksi
smooth_left = {}
smooth_right = {}
hand_velocity = 0.0
ALPHA = 0.6  

# State untuk Satu Tangan (Ganti manual via kepalan tangan)
color_index = 0
effect_index = 0
is_fisted_prev = False  

# Palet Pilihan Warna (Format BGR)
COLORS = [
    (0, 0, 255),    # 0: Merah
    (255, 0, 0),    # 1: Biru
    (0, 255, 255),  # 2: Kuning
    (0, 255, 0),    # 3: Hijau
    (255, 0, 255),  # 4: Magenta
    (255, 128, 0),  # 5: Oranye
]

# Daftar Efek Sisi
EFFECT_MODES = ['normal', 'mono', 'negative', 'sepia', 'bright']

# Penyimpanan riwayat posisi untuk efek ekor komet
trail_history_left = {i: deque(maxlen=15) for i in range(5)}
trail_history_right = {i: deque(maxlen=15) for i in range(5)}

# Membuka kamera web
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Tidak dapat mengakses kamera! Pastikan tidak ada aplikasi lain yang sedang menggunakan webcam.")
    exit()

# Mengatur resolusi kamera ke rasio 4:3 (640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Fungsi EMA Smoothing
def apply_smoothing(current_dict, smooth_dict):
    updated_dict = {}
    for i in current_dict:
        curr_x, curr_y = current_dict[i]
        if i in smooth_dict:
            prev_x, prev_y = smooth_dict[i]
            smooth_x = int(ALPHA * curr_x + (1 - ALPHA) * prev_x)
            smooth_y = int(ALPHA * curr_y + (1 - ALPHA) * prev_y)
            updated_dict[i] = (smooth_x, smooth_y)
        else:
            updated_dict[i] = (curr_x, curr_y)
    return updated_dict

# Deteksi apakah tangan sedang mengepal
def check_is_fisted(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x * w, lm[0].y * h])
    
    distances = []
    for tip_idx in FINGER_TIPS:
        tip = np.array([lm[tip_idx].x * w, lm[tip_idx].y * h])
        distances.append(np.linalg.norm(tip - wrist))
    
    avg_distance = np.mean(distances)
    return avg_distance < 100.0

# Fungsi gelombang dinamis saat tangan melambai
def apply_motion_wave_effect(pts_list, velocity, time_val):
    wavy_pts = []
    amplitude = min(max(velocity * 0.4, 0.0), 12.0)
    
    if amplitude < 0.5:
        return pts_list  

    for idx, (x, y) in enumerate(pts_list):
        offset_x = int(amplitude * np.sin(time_val * 12 + idx * 1.5))
        offset_y = int(amplitude * np.cos(time_val * 12 + idx * 1.5))
        wavy_pts.append((x + offset_x, y + offset_y))
    return wavy_pts

# Fungsi untuk menerapkan efek visual secara cepat
def apply_patch_effect(patch, effect_type, tint_color):
    if effect_type == 'mono':
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif effect_type == 'negative':
        processed = cv2.bitwise_not(patch)
    elif effect_type == 'sepia':
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(patch, sepia_kernel)
        processed = np.clip(sepia_img, 0, 255).astype(np.uint8)
    elif effect_type == 'bright':
        processed = cv2.convertScaleAbs(patch, alpha=1.3, beta=25)
    else:  # 'normal'
        processed = patch

    colored_overlay = cv2.addWeighted(processed, 0.75, np.full_like(processed, tint_color), 0.25, 0)
    return colored_overlay

# Fungsi menggambar ekor komet dengan gradasi halus (fade ke ujung) menggunakan overlay transparan
def draw_smooth_comet_trails(frame, trail_history, tip_to_color_map):
    overlay = frame.copy()
    
    for tip_idx, history in trail_history.items():
        points = list(history)
        num_pts = len(points)
        if num_pts < 3:
            continue
        
        trail_color = tip_to_color_map.get(tip_idx, (0, 255, 255))
        
        for i in range(num_pts - 1):
            pt1 = points[i]     
            pt2 = points[i+1]   
            
            progress = i / (num_pts - 1)
            thickness = int(max(1, 6 * (1.0 - progress)))
            
            cv2.line(overlay, pt1, pt2, trail_color, thickness, cv2.LINE_AA)
            
            if i == 0:
                cv2.circle(overlay, pt1, 4, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

print("="*60)
print(" PROGRAM GEOMETRI TANGAN DIAKTIFKAN (4:3 RATIO & SMOOTH FADE) ")
print(" - Resolusi kamera diset ke 4:3 (640x480)")
print(" - Ekor komet halus, mengalir, dan memudar (fade) ke ujung")
print(" - Tekan tombol 'q' untuk keluar program")
print("="*60)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("PERINGATAN: Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_left = {}
        current_right = {}
        is_current_fisted = False
        num_hands_detected = 0

        if results.multi_hand_landmarks is not None:
            num_hands_detected = len(results.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                actual_hand = "Right" if label == "Left" else "Left"
                
                if num_hands_detected == 1:
                    is_current_fisted = check_is_fisted(hand_landmarks, w, h)

                lm = hand_landmarks.landmark
                fingertip_dict = {}
                for i, tip_idx in enumerate(FINGER_TIPS):
                    cx, cy = int(lm[tip_idx].x * w), int(lm[tip_idx].y * h)
                    fingertip_dict[i] = (cx, cy)
                    
                    if actual_hand == "Left":
                        trail_history_left[i].appendleft((cx, cy))  
                    else:
                        trail_history_right[i].appendleft((cx, cy))
                    
                if actual_hand == "Left": 
                    current_left = fingertip_dict
                else: 
                    current_right = fingertip_dict
        else:
            for i in range(5):
                trail_history_left[i].clear()
                trail_history_right[i].clear()

        # Logika Pergantian Warna & Efek Manual via Kepalan Tangan (Satu Tangan)
        if num_hands_detected == 1 and is_current_fisted and not is_fisted_prev:
            color_index = (color_index + 1) % len(COLORS)
            effect_index = (effect_index + 1) % len(EFFECT_MODES)
        is_fisted_prev = is_current_fisted

        smooth_left = apply_smoothing(current_left, smooth_left)
        smooth_right = apply_smoothing(current_right, smooth_right)
        if not current_left: smooth_left = {}
        if not current_right: smooth_right = {}

        # Hitung kecepatan gerakan tangan untuk gelombang
        current_centers = []
        if smooth_left:
            current_centers.append(np.mean(list(smooth_left.values()), axis=0))
        if smooth_right:
            current_centers.append(np.mean(list(smooth_right.values()), axis=0))
        
        if current_centers:
            curr_avg_center = np.mean(current_centers, axis=0)
            if 'prev_avg_center' in locals():
                hand_velocity = np.linalg.norm(curr_avg_center - prev_avg_center)
            else:
                hand_velocity = 0.0
            prev_avg_center = curr_avg_center
        else:
            hand_velocity = 0.0

        t_now = time.time()
        output_frame = frame.copy()

        tip_to_color_map = {}
        mode_desc = "Menunggu Tangan..."  

        # --- Skenario A: Hanya Satu Tangan Aktif ---
        active_hand_data = None
        if len(smooth_left) > 0 and len(smooth_right) == 0:
            active_hand_data = smooth_left
        elif len(smooth_right) > 0 and len(smooth_left) == 0:
            active_hand_data = smooth_right

        if active_hand_data:
            active_color = COLORS[color_index]
            active_effect = EFFECT_MODES[effect_index]
            mode_desc = f"Satu Tangan | Warna: {color_index+1} | Efek: {active_effect.upper()}"
            
            for i in active_hand_data.keys():
                tip_to_color_map[i] = active_color

            sorted_indices = sorted(list(active_hand_data.keys()))
            pts_list = [active_hand_data[i] for i in sorted_indices]
            
            if len(pts_list) >= 3:
                wavy_pts = apply_motion_wave_effect(pts_list, hand_velocity, t_now)
                pts_np = np.array(wavy_pts, dtype=np.int32)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts_np], 255)
                
                processed_patch = apply_patch_effect(frame, active_effect, active_color)
                np.copyto(output_frame, processed_patch, where=(mask[:, :, None] == 255))
                cv2.polylines(output_frame, [pts_np], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # --- Skenario B: Kedua Tangan Aktif (4 Sisi Berbeda Warna & Efek) ---
        elif len(smooth_left) > 0 and len(smooth_right) > 0:
            mode_desc = "Dua Tangan (Multi-Color & Multi-Effect 4 Sisi)"
            available_indices = [i for i in range(5) if i in smooth_left and i in smooth_right]
            
            side_palette = [
                (COLORS[0], 'mono'),      # Sisi 1: Merah + Monokrom
                (COLORS[1], 'negative'),  # Sisi 2: Biru + Negatif
                (COLORS[2], 'sepia'),     # Sisi 3: Kuning + Sepia
                (COLORS[3], 'bright')     # Sisi 4: Hijau + Bright
            ]
            
            if len(available_indices) >= 2:
                for idx in range(len(available_indices) - 1):
                    i1 = available_indices[idx]
                    i2 = available_indices[idx + 1]
                    
                    if i1 in smooth_left and i2 in smooth_left and i1 in smooth_right and i2 in smooth_right:
                        p_L1 = smooth_left[i1]
                        p_L2 = smooth_left[i2]
                        p_R2 = smooth_right[i2]
                        p_R1 = smooth_right[i1]
                        
                        quad_raw = [p_L1, p_L2, p_R2, p_R1]
                        wavy_quad = apply_motion_wave_effect(quad_raw, hand_velocity, t_now + idx * 0.2)
                        quad_pts = np.array(wavy_quad, dtype=np.int32)
                        
                        mask_side = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask_side, [quad_pts], 255)
                        
                        side_color, side_effect = side_palette[idx % len(side_palette)]
                        
                        tip_to_color_map[i1] = side_color
                        tip_to_color_map[i2] = side_color
                        
                        processed_patch = apply_patch_effect(frame, side_effect, side_color)
                        np.copyto(output_frame, processed_patch, where=(mask_side[:, :, None] == 255))
                        cv2.polylines(output_frame, [quad_pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Gambar efek ekor komet yang halus dan memudar (fade)
        draw_smooth_comet_trails(output_frame, trail_history_left, tip_to_color_map)
        draw_smooth_comet_trails(output_frame, trail_history_right, tip_to_color_map)

        # Tampilkan Status di Layar
        cv2.putText(output_frame, mode_desc, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('4:3 Hand Geometry & Smooth Comet', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Terjadi kesalahan saat menjalankan program: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()