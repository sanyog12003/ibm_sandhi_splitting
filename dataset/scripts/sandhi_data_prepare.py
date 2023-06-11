from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import devnagri_reader as dr
import numpy as np
from sklearn.model_selection import train_test_split

swaras = ['a', 'A', 'i', 'I', 'u', 'U', 'e', 'E', 'o', 'O', 'f', 'F', 'x', 'X']
vyanjanas = ['k', 'K', 'g', 'G', 'N', 
             'c', 'C', 'j', 'J', 'Y',
             'w', 'W', 'q', 'Q', 'R',
             't', 'T', 'd', 'D', 'n',
             'p', 'P', 'b', 'B', 'm',
             'y', 'r', 'l', 'v','S', 'z', 's', 'h', 'L', '|']
others = ['H', 'Z', 'V', 'M', '~', '/', '\\', '^', '\'', ' ']

slp1charlist = swaras + vyanjanas + others

def remove_nonslp1_chars(word):
    newword = ''
    for char in word:
        if char in slp1charlist:
            newword = newword + char
    return newword

def get_sandhi_dataset(datafile, keep_all=False):
    datalist = []
    datalist_single_split = []

    with open(datafile) as fp:
        tests = fp.read().splitlines()
    
    total = 0
    maxlen = 0
    with open("input.txt", "w") as fwrite_input, open("output.txt", "w") as fwrite_output, open("input_single_split.txt", "w") as fwrite_input_single_split, open("output_single_split.txt", "w") as fwrite_output_single_split:
        for test in tests:
            if(test.find('=>') == -1):
                continue
            inout = test.split('=>')
            local_datalist = []
            local_datalist_single_split = []
            if(len(inout[0]) <= len(inout[1]) and len(inout[1]) > 1 and len(inout[0]) > 1):
                expected = inout[0].strip()
                expected = dr.read_devnagri_text(expected)
                slp1expected = transliterate(expected, sanscript.DEVANAGARI, sanscript.SLP1)
                slp1expected = remove_nonslp1_chars(slp1expected)

                fwrite_input.write("{}\n".format(slp1expected))
                local_datalist.append(slp1expected)

                words = inout[1].split('+')
                word_trans = []
                for word in words:
                    temp = dr.read_devnagri_text(word)
                    temp = transliterate(temp, sanscript.DEVANAGARI, sanscript.SLP1)
                    temp = remove_nonslp1_chars(temp)
                    word_trans.append(temp)
                slp1words = '+'.join(word_trans)
                fwrite_output.write("{}\n".format(slp1words))

                if len(words) == 2:
                    fwrite_input_single_split.write("{}\n".format(slp1expected))
                    local_datalist_single_split.append(slp1expected)
                    local_datalist_single_split.append(slp1words)
                    fwrite_output_single_split.write("{}\n".format(slp1words))
                local_datalist.append(slp1words)
            else:
                continue

            datalist.append(local_datalist)
            if len(local_datalist_single_split) != 0:
                datalist_single_split.append(local_datalist_single_split)

    """
    print("len(datalist) :{}".format(len(datalist)))
    dtrain, dtest = train_test_split(datalist, test_size=0.2, random_state=1)
    dtrain, dvalid = train_test_split(dtrain, test_size=0.1, random_state=1)
    dtrain_single_split, dtest_single_split = train_test_split(datalist_single_split, test_size=0.2, random_state=1)
    dtrain_single_split, dvalid_single_split = train_test_split(dtrain_single_split, test_size=0.1, random_state=1)

    with open("train.src", "w") as fwrite_input, open("train.tgt", "w") as fwrite_output:
        for i in range(len(dtrain)):
            fwrite_input.write("{}\n".format(dtrain[i][0]))
            fwrite_output.write("{}\n".format(dtrain[i][1]))
    with open("test.src", "w") as fwrite_input, open("test.tgt", "w") as fwrite_output:
        for i in range(len(dtest)):
            fwrite_input.write("{}\n".format(dtest[i][0]))
            fwrite_output.write("{}\n".format(dtest[i][1]))
    with open("valid.src", "w") as fwrite_input, open("valid.tgt", "w") as fwrite_output:
        for i in range(len(dvalid)):
            fwrite_input.write("{}\n".format(dvalid[i][0]))
            fwrite_output.write("{}\n".format(dvalid[i][1]))
    with open("train_single_split.src", "w") as fwrite_input, open("train_single_split.tgt", "w") as fwrite_output:
        for i in range(len(dtrain_single_split)):
            fwrite_input.write("{}\n".format(dtrain_single_split[i][0]))
            fwrite_output.write("{}\n".format(dtrain_single_split[i][1]))
    with open("test_single_split.src", "w") as fwrite_input, open("test_single_split.tgt", "w") as fwrite_output:
        for i in range(len(dtest_single_split)):
            fwrite_input.write("{}\n".format(dtest_single_split[i][0]))
            fwrite_output.write("{}\n".format(dtest_single_split[i][1]))
    with open("valid_single_split.src", "w") as fwrite_input, open("valid_single_split.tgt", "w") as fwrite_output:
        for i in range(len(dvalid_single_split)):
            fwrite_input.write("{}\n".format(dvalid_single_split[i][0]))
            fwrite_output.write("{}\n".format(dvalid_single_split[i][1]))
    """

"""
def generateFiles(input_file, output_file):
    with open(input_file) as fread:
        lines = fread.readlines()
    with open("train.src", "w") as fwrite_train, open("test.src", "w") as fwrite_test:
        for index, line in enumerate(lines):
            if((index+1)%5 == 0):
                fwrite_test.write(line)
            else:
                fwrite_train.write(line)
    with open("train.src") as fread:
        lines = fread.readlines()
    with open("train.src", "w") as fwrite_train, open("valid.src", "w") as fwrite_valid:
        for index, line in enumerate(lines):
            if((index+1)%5 == 0):
                fwrite_valid.write(line)
            else:
                fwrite_train.write(line)
                
    with open(output_file) as fread:
        lines = fread.readlines()
    with open("train.tgt", "w") as fwrite_train, open("test.tgt", "w") as fwrite_test:
        for index, line in enumerate(lines):
            if((index+1)%5 == 0):
                fwrite_test.write(line)
            else:
                fwrite_train.write(line)
    with open("train.tgt") as fread:
        lines = fread.readlines()
    with open("train.tgt", "w") as fwrite_train, open("valid.tgt", "w") as fwrite_valid:
        for index, line in enumerate(lines):
            if((index+1)%5 == 0):
                fwrite_valid.write(line)
            else:
                fwrite_train.write(line)
"""

def get_xy_data(datafile):
    get_sandhi_dataset(datafile)
    #generateFiles("input.txt", "output.txt")
    #return w1l, w2l, ol
    
get_xy_data("sandhiset.txt")
