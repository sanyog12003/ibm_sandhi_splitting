import os
import glob, sys

with open("uoh.txt", "r") as fread:
    lines = fread.readlines()

incorrect_splits = 0
total_splits = len(lines)
for line in lines:
    columns = line.strip().split(",")
    if(len(columns[0]) - 2 < len(columns[1])):
        continue
    else:
        incorrect_splits = incorrect_splits + 1
        print(columns[0], columns[1])

print("incorrect_splits : {}/{} = {}".format(incorrect_splits, total_splits, incorrect_splits/total_splits * 100))
        
