import glob
import os
from os import listdir, getcwd
dirs = ['blue_back_mixed', 'blue_background', 'crabClaw', 'pincer', 'rotating', 'rotating_no_light', 'scissor_golden', 'scissor_length1', 'scissor_length2', 'shovel']
counts=[15, 10, 359, 359, 10, 17, 359, 359, 359, 359]
cwd = getcwd()

for i in range(len(dirs)):
    dir_path = dirs[i]
    current_dir = cwd + '/' + dir_path# PATH TO IMAGE DIRECTORY
# Percentage of images to be used for the valid set
    percentage_test = 30;
# Create train.txt and valid.txt
    file_train = open('train.txt', 'w')
    file_test = open('valid.txt', 'w')
# Populate train.txt and valid.txt
    counter = 1
    index_test = round(30/100*counts[i])
    for file in glob.iglob(os.path.join(current_dir, '*.jpg')):
        title, ext = os.path.splitext(os.path.basename(file))
        if counter <= index_test:
            #counter = 1
            file_test.write(cwd + "/" + title + '.jpg' + "\n")
        else:
            file_train.write(cwd + "/" + title + '.jpg' + "\n")
        counter = counter + 1
