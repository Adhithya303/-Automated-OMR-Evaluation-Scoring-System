import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import os
import io
import tempfile

# ---------------- Helper Functions ----------------

def load_and_parse_answer_key(file):
    """Parses CSV into answer key dict."""
    try:
        df = pd.read_csv(file)
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
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None

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

def evaluate_sheet(image_bytes, answer_key):
    """Evaluate a single OMR sheet."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 4
    )

    # --- simplified: skip perspective warp if fiducials not detected ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = [
        c for c in contours
        if CONFIG["BUBBLE_MIN_WIDTH"] <= cv2.boundingRect(c)[2] <= CONFIG["BUBBLE_MAX_WIDTH"]
        and CONFIG["BUBBLE_MIN_HEIGHT"] <= cv2.boundingRect(c)[3] <= CONFIG["BUBBLE_MAX_HEIGHT"]
    ]
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
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
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

# ---------------- Streamlit UI ----------------

st.title("📄 Automated OMR Evaluation System")

st.sidebar.header("Upload Files")
answer_key_file = st.sidebar.file_uploader("Upload Answer Key (CSV)", type=["csv"])
uploaded_images = st.sidebar.file_uploader("Upload OMR Sheets", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if answer_key_file and uploaded_images:
    answer_key = load_and_parse_answer_key(answer_key_file)
    if answer_key is None:
        st.error("Invalid answer key.")
    else:
        results = []
        for img_file in uploaded_images:
            score = evaluate_sheet(img_file, answer_key)
            if score is not None:
                results.append({"Filename": img_file.name, "Score": score})

        if results:
            df = pd.DataFrame(results)
            st.success(f"✅ Processed {len(results)} sheets")
            st.dataframe(df)

            # Download link for results
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", data=csv_bytes, file_name="final_scores.csv", mime="text/csv")
        else:
            st.warning("No sheets processed.")
else:
    st.info("Please upload an answer key CSV and at least one OMR image.")
