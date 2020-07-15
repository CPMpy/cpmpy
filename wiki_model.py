#!/usr/bin/python3
"""
Logic grid puzzle: 'origin' in CPpy

Based on... to check originally, currently part of ZebraTutor
Probably part of Jens Claes' master thesis, from a 'Byron...' booklet
"""
# import sys
# sys.path.append('/home/crunchmonster/Documents/VUB/01_SharedProjects/01_cppy_src')
from cppy import *
from pathlib import Path
import re
# import numpy
# import pandas as pd

model2 = Model([ implies( ((BoolVar() | BoolVar()) & BoolVar()) , ~BoolVar() )   ])
print(model2)
cnf2, new_vars = model2.to_cnf()
print(new_vars)
# for i, formula in enumerate(cnf2):
#     print(i, formula, formula.to_cnf())
print("\nCNF:")
print(cnf2)

## VOCABULARY
# base pattern for vocabulary structure
pattern_voc = "vocabulary.*?\{(.|\n)*?\n\}"

# STRUCTURE Pattern
pattern_structure = "structure.*?\{(.|\n)*?\n\}"
# pattern_structure = "structure\s+\w+\s*:\s*\w*\s*\{(.|\n)*?\n\}"

## THEORIES
pattern_theory = "theory.*?\{(.|\n)*?\n\}"

## PROCEDURE
pattern_procedure = "procedure.*?\{(.|\n)*?\n\}"

## COMMENT LINE
pattern_comment = "\/\/(.+)"

## FUNCTION CALL 
pattern_function='\w+\(.*\)'

def get_voc_types(vocabulary, voc_data):

    types= {}
    relations = []
    for line in vocabulary.split('\n'):
        line = line.strip()

        # ignore comment line
        if line.startswith("//"):
            continue

        # line starts with type
        if line.startswith("type"):
            line = line.replace('type', '', 1).strip()
            typename, typedescription = line.split(' ', 1)
            if 'constructed from' in typedescription:
                sGroup = re.search("\{.*\}", typedescription).group().replace('{','').replace('}','')
                types[typename] = [elem.strip() for elem in re.split(";|,", sGroup)]
            elif '=' in typedescription and "int" in typedescription :
                sGroup = re.search("\{.*\}", typedescription).group().replace('{','').replace('}','')
                # type assumptiontype = {0..36} isa int
                if '..' in sGroup:
                    splitted = sGroup.split('..')
                    if len(splitted) > 2:
                        raise f"Not handled {sGroup}"
                    start, end = int(splitted[0]), int(splitted[1])
                    types[typename] = [str(i) for i in range(start, end+1)]
                # type dollar = {5; 6; 7; 8; 9} isa int
                else:
                    types[typename] = [elem.strip() for elem in re.split(";|,", sGroup)]
            elif re.search(pattern_function, line) != None:
                # relation between two types
                rel = re.search(pattern_function, line).group()

            else:
                raise f"Not supported : {line}"
    voc_data['types'] = types
    return voc_data

def get_theories(theories, data):
    data['clues'] = []
    data['bij'] = []
    data['trans'] = []

    return data

def read_idp(filename):

    f = open(filename, 'r')
    original_text = f.read()
    text = original_text.strip()

    # text = line 
    comments = []
    for match in re.finditer(pattern_comment, text):
        # Complete match (string)
        comment = match.group()
        comments.append(comment)

    # match vocabulary
    vocabulary = []
    for match in re.finditer(pattern_voc, text):
        # Complete match (string)
        voc = match.group()
        vocabulary.append(voc)

    voc_data = {}
    for voc in vocabulary:
        voc_data = get_voc_types(voc, voc_data)

        print(data_voc)
        
    theories = []
    #https://riptutorial.com/python/example/12097/iterating-over-matches-using--re-finditer-
    for match in re.finditer(pattern_theory, text):
        # Complete match (string)
        theory = match.group()
        theories.append(theory)

    procedures = []
    #https://riptutorial.com/python/example/12097/iterating-over-matches-using--re-finditer-
    for match in re.finditer(pattern_procedure, text):
        # Complete match (string)
        proc = match.group()
        procedures.append(proc)

    f.close()

# filename = "/home/crunchmonster/Documents/VUB/01_SharedProjects/03_holygrail/experiments/03_OMUS/02_cppy/data/"
# filename += "p12.idp"
# read_idp(filename)
