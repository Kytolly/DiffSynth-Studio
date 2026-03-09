import os
import pandas as pd

dataset_root = os.path.join("FollowBench/train", "")
data = []

PROMPT = 'Transform it into the third-person perspective.'

for case_folder in sorted(os.listdir(dataset_root)):
    if not case_folder.startswith("train_case_"):
        continue
    
    case_path = os.path.join(dataset_root, case_folder)
    video_path = os.path.join(case_folder, "exo.mp4") 
    vace_video_path = os.path.join(case_folder, "ego.mp4")
    vace_ref_image_path = os.path.join(case_folder, "ref_img.jpg")
    data.append({
        "video": video_path,
        "vace_video": vace_video_path,
        "vace_reference_image": vace_ref_image_path,
        "prompt": PROMPT
    })

df = pd.DataFrame(data)
csv_path = os.path.join(dataset_root, "metadata_vace.csv")
df.to_csv(csv_path, index=False)
print(f"成功生成元数据文件：{csv_path}")