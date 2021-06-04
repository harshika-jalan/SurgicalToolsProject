#@source-https://github.com/theAIGuysCode/YOLOv3-Cloud-Tutorial
import os

image_files = []
index=0
os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data/obj/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
        index = index+1
        if index == 1547:
            break
    outfile.close()
os.chdir("..")
