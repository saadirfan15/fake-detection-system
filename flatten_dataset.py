import os, shutil

src = r'D:\computer vision project\dataset\PKR_real'

for folder in os.listdir(src):
    folder_path = os.path.join(src, folder)
    if os.path.isdir(folder_path):
        for img in os.listdir(folder_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                new_name = f"{folder}_{img}"
                shutil.move(
                    os.path.join(folder_path, img),
                    os.path.join(src, new_name)
                )
        os.rmdir(folder_path)

print("Done! All images moved to PKR_real/")