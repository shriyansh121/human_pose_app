import os

# ✅ All paths are now RELATIVE to the current project folder
FOLDERS = [
    "src",
    "data",
    "data/input_videos",
    "data/output_reports",
    "models"
]

FILES = {
    "app.py": "",
    "requirements.txt": "",
    "src/squat.py": "",
    "src/utils.py": "",
    "models/.gitkeep": ""
}

def create_project_components():
    print("\n🚀 Adding project components inside existing folder...\n")

    # ✅ Create folders
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Folder created: {folder}")

    # ✅ Create files
    for filepath, content in FILES.items():
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)
            print(f"✅ File created: {filepath}")
        else:
            print(f"⚠️ Already exists: {filepath}")

    print("\n🎯 All components added successfully!")

if __name__ == "__main__":
    create_project_components()
