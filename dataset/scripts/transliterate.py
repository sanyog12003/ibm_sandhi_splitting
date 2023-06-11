from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import devnagri_reader as dr
import sys, os, glob

text1 = sys.argv[1]
text2 = sys.argv[2]
if ".txt" in text1:
    if(".txt" in text2):
        flag = False
    else:
        raise Exception("Both shoule be text file.")
else:
    if(".txt" in text2):
        raise Exception("Both needs to be of same category - either text files or samples")
    else:
        flag = True
    

if flag:
    source_text1 = text1.strip()
    source_text1 = transliterate(source_text1, sanscript.SLP1, sanscript.DEVANAGARI)
    source_text2 = ""
    for comp in text2.strip().split("\n")[0].split("+"):
        comp = transliterate(comp, sanscript.SLP1, sanscript.DEVANAGARI)
        source_text2 = source_text2 + comp
else:
    with open("../"+text1, "r") as fread1:
        lines1 = fread1.readlines()
    with open("../" + text2, "r") as fread2:
        lines2 = fread2.readlines()
    for line1, line2 in zip(lines1, lines2):
        columns = line1.strip().split(",")
        source_text1 = transliterate(columns[0], sanscript.SLP1, sanscript.DEVANAGARI)
        source_text2 = ""
        for comp in columns[1].strip().split("\n")[0].split("+"):
            temp = transliterate(comp, sanscript.SLP1, sansscript.DEVANAGRUi)
            source_text2 = source_text2 + comp
print("{} \t {}".format(source_text1, source_text2))
            
