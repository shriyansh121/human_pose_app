import streamlit as st
import os
from src.squat.squat import analyze_squat

st.set_page_config(page_title="Pose Analyzer AI", layout="wide")

st.title("🏋️ AI Pose Analyzer")
st.markdown("Upload your workout video and receive a full AI performance report!")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

# ✅ FUNCTION TO CREATE MATCHING SESSION FOLDERS
def create_new_session_folders(
    input_base="data/input_videos",
    output_base="data/output_reports"
):
    os.makedirs(input_base, exist_ok=True)
    os.makedirs(output_base, exist_ok=True)

    existing = [f for f in os.listdir(output_base) if f.startswith("session_")]
    session_num = len(existing) + 1

    session_name = f"session_{session_num}"

    input_session_dir = os.path.join(input_base, session_name)
    output_session_dir = os.path.join(output_base, session_name)

    os.makedirs(input_session_dir, exist_ok=True)
    os.makedirs(output_session_dir, exist_ok=True)

    return input_session_dir, output_session_dir


if uploaded_file:

    if st.button("🚀 Analyze My Exercise"):
        with st.spinner("Analyzing form..."):

            # ✅ CREATE MATCHING SESSION FOLDERS
            input_session_dir, output_session_dir = create_new_session_folders()

            # ✅ SAVE INPUT VIDEO INSIDE ITS OWN SESSION FOLDER
            video_path = os.path.join(input_session_dir, uploaded_file.name)

            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            # ✅ PASS OUTPUT SESSION FOLDER TO BACKEND
            pdf_path, reps = analyze_squat(
                input_video_path=video_path,
                output_dir=output_session_dir
            )

        st.success("✅ Analysis Complete!")

        st.metric("Total Reps", len(reps))

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Full Report",
                f,
                file_name=os.path.basename(pdf_path),
            )

        st.info(f"📂 Input saved in: {input_session_dir}")
        st.info(f"📂 Output saved in: {output_session_dir}")
