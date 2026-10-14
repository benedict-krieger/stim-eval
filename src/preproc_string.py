import re

def preproc_string(in_string):
    '''
    Preprocess string so it contains single whitespace between words and non-alphanumeric characters.
    
    in_string: un-preprocessed string
    
    Returns preprocessed string.

    '''
    
    regexp1 = re.compile("(?<=[^\s])(?=[\W])|(?<=[\W])(?=[^\s])") # Lb: non-whitespace La: [.,] "schön." | Lb: [.,] La: non-whitespace ".schön"
    regexp2 = re.compile("\s\s+") # reduce multiple whitespaces
    
    new_string = regexp1.sub(" ",in_string)
    new_string = regexp2.sub(" ",new_string)
    
    return new_string
