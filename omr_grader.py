import os
import sys
import cv2
import numpy as np
import pandas as pd
import re
import glob

def get_jpeg_images(folder_path):
    """
    Scans a given folder and returns a list of all JPEG image filenames.
    """
    if not os.path.isdir(folder_path):
        return []
    
    jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpeg', '.jpg'))]
    return [f for f in jpeg_files if os.path.isfile(os.path.join(folder_path, f))]

def find_csv_file(folder_path):
    """
    Scans a given folder for a single CSV file and returns its full path.
    Returns None if no CSV is found or if multiple are present.
    """
    # Use glob to find all files ending with .csv in the specified directory
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    # Check if exactly one CSV file was found
    if len(csv_files) == 1:
        return csv_files[0]
    return None

def load_and_parse_answer_key(csv_path):
    """
    Loads and parses the specific CSV format into a standard answer key dictionary.
    """
    if not os.path.exists(csv_path):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        answers_by_q_num = {}
        for column in df.columns:
            for cell_value in df[column].dropna():
                cleaned_str = str(cell_value).strip()
                parts = re.split(r'\s*[-.]\s*', cleaned_str)
                
                if len(parts) == 2:
                    q_num_str, ans_letter = parts
                    try:
                        q_num = int(q_num_str.strip())
                        answer = ans_letter.strip().upper()
                        answers_by_q_num[q_num] = answer
                    except ValueError:
                        pass
        
        sorted_q_nums = sorted(answers_by_q_num.keys())
        return {i: answers_by_q_num[q_num] for i, q_num in enumerate(sorted_q_nums)}
    except Exception:
        return None

# --- OMR Configuration ---
CONFIG = {
    "NUM_OPTIONS": 4,
    "NUM_QUESTIONS": 100,
    "ANSWER_MAP": {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
    "FIDUCIAL_MIN_AREA": 500,
    "FIDUCIAL_ASPECT_RATIO": 0.8,
    "BUBBLE_MIN_WIDTH": 15,
    "BUBBLE_MAX_WIDTH": 35,
    "BUBBLE_MIN_HEIGHT": 15,
    "BUBBLE_MAX_HEIGHT": 35,
    "MARK_THRESHOLD": 150
}

def evaluate_sheet(image_path, answer_key):
    """
    Processes a single OMR sheet, scores it, and returns the score.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fiducial_contours = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if len(approx) == 4 and area > CONFIG["FIDUCIAL_MIN_AREA"] and CONFIG["FIDUCIAL_ASPECT_RATIO"] <= aspect_ratio <= (1/CONFIG["FIDUCIAL_ASPECT_RATIO"]):
            fiducial_contours.append(approx)

    if len(fiducial_contours) != 4:
        warped_thresh = thresh
    else:
        fiducial_points = np.concatenate(fiducial_contours).reshape(-1, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = fiducial_points.sum(axis=1)
        rect[0] = fiducial_points[np.argmin(s)]
        rect[2] = fiducial_points[np.argmax(s)]
        diff = np.diff(fiducial_points, axis=1)
        rect[1] = fiducial_points[np.argmin(diff)]
        rect[3] = fiducial_points[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_thresh = cv2.warpPerspective(thresh, M, (maxWidth, maxHeight))

    contours, _ = cv2.findContours(warped_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_contours = [c for c in contours if CONFIG["BUBBLE_MIN_WIDTH"] <= cv2.boundingRect(c)[2] <= CONFIG["BUBBLE_MAX_WIDTH"] and CONFIG["BUBBLE_MIN_HEIGHT"] <= cv2.boundingRect(c)[3] <= CONFIG["BUBBLE_MAX_HEIGHT"]]
    bubble_contours.sort(key=lambda c: cv2.boundingRect(c)[1])

    questions = []
    for i in range(0, len(bubble_contours), CONFIG["NUM_OPTIONS"]):
        chunk = bubble_contours[i:i + CONFIG["NUM_OPTIONS"]]
        if len(chunk) == CONFIG["NUM_OPTIONS"]:
            chunk.sort(key=lambda c: cv2.boundingRect(c)[0])
            questions.append(chunk)

    student_answers = {}
    score = 0

    for q_index, question_options in enumerate(questions):
        if q_index >= CONFIG["NUM_QUESTIONS"]:
            break

        marked_bubble_index = -1
        max_filled = 0
        
        for j, bubble in enumerate(question_options):
            mask = np.zeros(warped_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            mask = cv2.bitwise_and(warped_thresh, warped_thresh, mask=mask)
            filled_pixels = cv2.countNonZero(mask)

            if filled_pixels > max_filled:
                max_filled = filled_pixels
                marked_bubble_index = j
        
        if max_filled > CONFIG["MARK_THRESHOLD"]:
            student_ans_letter = CONFIG["ANSWER_MAP"].get(marked_bubble_index)
            student_answers[q_index] = student_ans_letter
            
    for q_num, correct_ans in answer_key.items():
        if student_answers.get(q_num) == correct_ans:
            score += 1
    
    return score

# --- Main script execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_image_folder>")
        sys.exit(1)

    image_folder = sys.argv[1]
    
    # Find the CSV file in the given directory
    key_file_path = find_csv_file(image_folder)

    if key_file_path is None:
        print("Error: No single CSV file found in the specified folder.")
        sys.exit(1)

    answer_key = load_and_parse_answer_key(key_file_path)
    
    if answer_key is None:
        print(f"Error: Could not load or parse the answer key from '{key_file_path}'. Exiting.")
        sys.exit(1)
        
    jpeg_images = get_jpeg_images(image_folder)
    
    if not jpeg_images:
        print(f"No JPEG images found in '{image_folder}'.")
        sys.exit(0)
        
    scores = []
    for image_name in jpeg_images:
        image_path = os.path.join(image_folder, image_name)
        score = evaluate_sheet(image_path, answer_key)
        if score is not None:
            scores.append({'Filename': image_name, 'Score': score})
    
    if scores:
        results_df = pd.DataFrame(scores)
        output_csv_path = os.path.join(image_folder, 'final_scores.csv')
        results_df.to_csv(output_csv_path, index=False)
        print(f"Successfully processed {len(scores)} sheets. Final scores saved to '{output_csv_path}'.")
    else:
        print("No sheets were successfully processed.")