import os,glob,sys

with open("uoh.txt", "r") as fread:
    lines = fread.readlines()

with open("uoh_single_split.txt", "w") as fwrite:
    for line in lines:
        splits = line.split(",")
        if(len(splits[1].split("+")) != 2):
            continue
        fwrite.write(line)

