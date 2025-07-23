# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

def create_precise_skin_mask(face_img, landmarks):
    """More precise skin mask - excluding eyes, nose, mouth"""
    face_h, face_w = face_img.shape[:2]
    available_landmarks = len(landmarks.landmark)
    
    # 1. Create full face mask
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    # Full face mask
    full_face_mask = np.zeros((face_h, face_w), dtype=np.uint8)
    
    oval_points = []
    for idx in FACE_OVAL:
        if idx < available_landmarks:
            x = int(landmarks.landmark[idx].x * face_w)
            y = int(landmarks.landmark[idx].y * face_h)
            oval_points.append([x, y])
    
    if len(oval_points) > 0:
        oval_points = np.array(oval_points)
        cv2.fillPoly(full_face_mask, [oval_points], 255)
    
    # 2. Create exclude regions mask
    exclude_mask = np.zeros((face_h, face_w), dtype=np.uint8)
    
    # Left eye region
    LEFT_EYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]
    
    # Right eye region
    RIGHT_EYE = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]
    
    # Nose region
    NOSE = [
        1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242
    ]
    
    # Mouth region
    MOUTH = [
        61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 
        402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82
    ]
    
    # Each region added to exclude mask
    for region_name, region_indices in [("eyes", LEFT_EYE + RIGHT_EYE), 
                                       ("nose", NOSE), 
                                       ("mouth", MOUTH)]:
        region_points = []
        for idx in region_indices:
            if idx < available_landmarks:
                x = int(landmarks.landmark[idx].x * face_w)
                y = int(landmarks.landmark[idx].y * face_h)
                region_points.append([x, y])
        
        if len(region_points) > 2:
            region_points = np.array(region_points)
            hull = cv2.convexHull(region_points)
            cv2.fillPoly(exclude_mask, [hull], 255)
    
    # Slightly expand exclude regions
    kernel = np.ones((15,15), np.uint8)
    exclude_mask = cv2.dilate(exclude_mask, kernel, iterations=1)
    
    # 3. Pure skin mask = full face - exclude regions
    skin_mask = cv2.bitwise_and(full_face_mask, cv2.bitwise_not(exclude_mask))
    
    # 4. Mask post-processing
    kernel = np.ones((5,5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Smooth edges
    skin_mask = cv2.GaussianBlur(skin_mask, (3,3), 0)
    _, skin_mask = cv2.threshold(skin_mask, 127, 255, cv2.THRESH_BINARY)
    
    return skin_mask

def test_precise_mask():
    """Test precise mask"""
    image_path = r"C:\Users\user\Desktop\Github\cosrec\KakaoTalk_20250715_141326156.jpg"
    
    # Existing function usage
    import face_detect
    face_img, basic_mask = face_detect.extract_face_with_skin_mask(image_path)
    
    if face_img is not None:
        # Re-run MediaPipe Face Mesh for precise mask
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                 refine_landmarks=True, min_detection_confidence=0.3) as face_mesh:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(face_rgb)
            
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0]
                precise_mask = create_precise_skin_mask(face_img, landmarks)
                
                # Compare results
                cv2.imshow("Original Face", face_img)
                cv2.imshow("Basic Mask", basic_mask)
                cv2.imshow("Precise Skin Mask", precise_mask)
                
                # Mask application results
                basic_result = cv2.bitwise_and(face_img, face_img, mask=basic_mask)
                precise_result = cv2.bitwise_and(face_img, face_img, mask=precise_mask)
                
                cv2.imshow("Basic Mask Applied", basic_result)
                cv2.imshow("Precise Mask Applied", precise_result)
                
                print("? Precise skin mask test completed")
                print("? Press ESC to exit")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    test_precise_mask()
