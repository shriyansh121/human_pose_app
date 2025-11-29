import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import calculate_angle
from src.save_report_video import save_full_report_and_video

#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()   # ✅ THIS LINE IS CRITICAL


def analyze_squat(input_video_path, output_dir):

    # --- Pose Estimation & Angle Calculation ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_angles = []
    frame_nums = []
    frames_for_key_images = []

    img_h, img_w = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if img_h == 0:
            img_h, img_w, _ = frame.shape

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            hip = [
                int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * img_w),
                int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * img_h),
            ]
            knee = [
                int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * img_w),
                int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * img_h),
            ]
            ankle = [
                int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img_w),
                int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img_h),
            ]

            angle = calculate_angle(hip, knee, ankle)

            if angle:
                frame_angles.append(angle)
                frame_nums.append(frame_num)
                frames_for_key_images.append(frame.copy())

    cap.release()
    pose.close()

    # --- 2. REP SEGMENTATION ---
    squat_threshold = 90
    reps = []
    current_rep = None
    in_squat = False

    for idx, angle in enumerate(frame_angles):
        if angle < squat_threshold and not in_squat:
            in_squat = True
            current_rep = {
                "start": frame_nums[idx],
                "min_angle": angle,
                "min_idx": idx,
            }

        elif angle < squat_threshold and in_squat:
            if angle < current_rep["min_angle"]:
                current_rep["min_angle"] = angle
                current_rep["min_idx"] = idx

        elif angle >= squat_threshold and in_squat:
            in_squat = False
            current_rep["end"] = frame_nums[idx]
            current_rep["duration"] = (
                current_rep["end"] - current_rep["start"]
            ) / fps
            reps.append(current_rep)
            current_rep = None

    # --- 3. PERFORMANCE METRICS ---
    avg_depth = np.mean([rep["min_angle"] for rep in reps]) if reps else 0
    fastest = min([rep["duration"] for rep in reps]) if reps else 0
    slowest = max([rep["duration"] for rep in reps]) if reps else 0
    std_depth = np.std([rep["min_angle"] for rep in reps]) if reps else 0
    session_time = (frame_nums[-1] - frame_nums[0]) / fps if frame_nums else 0

    os.makedirs(output_dir, exist_ok=True)

    # --- 4. PLOTS ---

    # -------- a. Angle Plot (Vibrant Line + Contrasting Threshold) --------
    plt.figure(figsize=(10, 5))

    # 🔴 Vibrant red-pink line for movement
    plt.plot(frame_nums, frame_angles, lw=2.5, color="#E11D48", label="Knee Angle")

    # 🟡 Golden-yellow threshold line
    plt.axhline(y=squat_threshold, linestyle="--", lw=2, color="#F59E0B", label="Squat Threshold")

    plt.xlabel("Frame")
    plt.ylabel("Knee Angle (Degrees)")
    plt.title("Squat Knee Angle Over Time", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)

    angle_plot_path = os.path.join(output_dir, "squat_angle_plot.png")
    plt.tight_layout()
    plt.savefig(angle_plot_path)
    plt.close()


    # -------- b. Rep Duration Bar Chart (Unique Color per Bar) --------
    rep_durations = [rep["duration"] for rep in reps]

    plt.figure(figsize=(7, 4))

    # 🎨 Unique vibrant color for each bar (no blue)
    bar_colors = [
        "#F97316",  # Orange
        "#22C55E",  # Green
        "#A855F7",  # Purple
        "#EF4444",  # Red
        "#EAB308",  # Yellow
        "#EC4899",  # Pink
        "#14B8A6",  # Teal
    ]

    plt.bar(
        range(1, len(rep_durations) + 1),
        rep_durations,
        color=bar_colors[:len(rep_durations)],
    )

    plt.xlabel("Repetition")
    plt.ylabel("Duration (seconds)")
    plt.title("Duration of Each Squat Rep", fontsize=12, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)

    rep_dur_plot_path = os.path.join(output_dir, "rep_duration_bar.png")
    plt.tight_layout()
    plt.savefig(rep_dur_plot_path)
    plt.close()


    # -------- c. Heatmap (Bright, High-Contrast Colormap) --------
    plt.figure(figsize=(7, 2.5))

    angles_array = np.array([rep["min_angle"] for rep in reps]).reshape(1, -1)

    # 🌋 Bright, high-contrast heatmap (NO BLUE)
    sns.heatmap(
        angles_array,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",     # Yellow → Orange → Red (very vibrant)
        linewidths=1,
        linecolor="black",
        cbar=True,
    )

    plt.yticks([])
    plt.xticks(
        ticks=np.arange(len(reps)) + 0.5,
        labels=[f"Rep {i+1}" for i in range(len(reps))],
        rotation=0,
    )

    plt.title("Minimum Knee Angle Per Rep (Squat Depth)", fontsize=12, fontweight="bold")

    heatmap_path = os.path.join(output_dir, "rep_angle_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()


    # --- 5. KEY FRAME IMAGES ---
    def save_pose_image(frame, pose_results, filename):
        annotated = frame.copy()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            annotated,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )
        cv2.imwrite(filename, annotated)

    key_frames_paths = {}
    frame_output_dir = os.path.join(output_dir, "key_frames")
    os.makedirs(frame_output_dir, exist_ok=True)

    if reps:
        min_rep_idx = min(range(len(reps)), key=lambda i: reps[i]["min_angle"])
        min_frame_idx = reps[min_rep_idx]["min_idx"]
        max_angle_idx = np.argmax(frame_angles)
        mid_rep_idx = len(reps) // 2
        mid_frame_idx = reps[mid_rep_idx]["min_idx"] if reps else 0

        pose_static = mp_pose.Pose(static_image_mode=True)

        for label, idx in [
            ("Deepest Squat", min_frame_idx),
            ("Standing", max_angle_idx),
            ("Mid Rep", mid_frame_idx),
        ]:
            fr = frames_for_key_images[idx]
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            results = pose_static.process(rgb)
            img_path = os.path.join(frame_output_dir, f"{label.replace(' ', '_').lower()}.png")
            save_pose_image(fr, results, img_path)
            key_frames_paths[label] = img_path

        pose_static.close()

    # --- 6. QUALITY ASSESSMENT ---
    good_reps = sum([1 for rep in reps if rep["min_angle"] <= 90])
    score = int(100 * good_reps / max(1, len(reps)))

    flags = []
    for rep in reps:
        if rep["min_angle"] > 105:
            flags.append(f"Rep {reps.index(rep) + 1}: Not deep enough")
        if rep["duration"] < 0.5:
            flags.append(f"Rep {reps.index(rep) + 1}: Too fast")

    # --- 7. GEMINI FEEDBACK ---
    llm = ChatOpenAI(
    model="gpt-4o-mini",   # Fast + cheap + very good for feedback
    temperature=0.3,
    max_retries=2,
    )

    prompt = f"""
    Analyze this squat exercise. The number of repetitions detected is {len(reps)}.
    The minimum angle recorded is {min(frame_angles):.1f}, and the maximum angle is {max(frame_angles):.1f}.

    Provide:
    - A professional, encouraging description of the performance.
    - Personalized suggestions to improve squat form.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    chat_response = (
        response.content if hasattr(response, "content") else str(response)
    )

    # --- ✅ FINAL SAVE CALL ---
    pdf_path = save_full_report_and_video(
        reps=reps,
        session_time=session_time,
        avg_depth=avg_depth,
        fastest=fastest,
        slowest=slowest,
        std_depth=std_depth,
        chat_response=chat_response,
        key_frames_paths=key_frames_paths,
        score=score,
        flags=flags,
        angle_plot_path=angle_plot_path,
        rep_dur_plot_path=rep_dur_plot_path,
        heatmap_path=heatmap_path,
        output_dir=output_dir,
    )

    return pdf_path, reps
