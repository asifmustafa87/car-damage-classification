# -*- coding: utf-8 -*-
import os

os.chdir("./dataset/raw_unsplit")

folder = os.getcwd()
print(folder)


# function that takes a folder name as input and outputs proportions of classes in that folder
def show_proportions(folder_x):
    # define all calsses in dictionary
    all_classes = ["scratch", "dent", "rim", "other"]
    summe = 0
    # iterate for all damage types
    for class_ in all_classes:
        os.chdir(f"{folder_x}/{class_}")
        # get current dir
        directory = os.getcwd()
        # your directory path
        lst = os.listdir(directory)
        number_files = len(lst)
        summe += number_files
        # print(class_,number_files)
    print("total number of images in folder: ", summe)
    for class_ in all_classes:
        os.chdir(f"{folder_x}/{class_}")
        # get current dir
        directory = os.getcwd()
        lst = os.listdir(directory)
        # your directory path
        number_files = len(lst)
        print(number_files)
        ratio = number_files / summe
        print(f"proportion of {class_} in folder: %.3f" % ratio)


# call function
show_proportions(folder)
