import sys

class InvalidBracketsException(Exception):
    pass

class InvalidNumberingException(Exception):
    pass

class InvalidAlignException(Exception):
    pass

def process_sen(sen):
    toks = []
    state = "normal"
    counter = 0
    for tok in sen.split():
        # checking valid brackets
        validate_brackets(tok)
        
        # processing
        if state == "normal":
            if tok == "[":
                state = "inside"
                actual_np = []
                continue
            else:
                toks.append(tok)
                continue
        elif state == "inside":
            if tok == "]":
                state = "after"
                continue
            else:
                actual_np.append(tok)
                continue
        elif state == "after":
            index = int(tok)
            # checking valid index ordering
            if index != counter:
                raise InvalidNumberingException
            
            counter += 1
            
            toks.append((index, actual_np))
            actual_np = []
            state = "normal"
            continue
    
    return toks
    
def validate_brackets(t):
    opener = t.find("[")
    closer = t.find("]")
    if opener > -1 or closer > -1:
        if len(t) > 1:
            sys.stderr.write("token: %s\n" % t)
            raise InvalidBracketsException
    

def extract_np_tok_indices(sen):
    indices = {}
    actual_index = 0
    for tok in sen:
        if isinstance(tok, tuple):
            index = tok[0]
            indices[index] = []
            for np_tok in tok[1]:
                indices[index].append(actual_index)
                actual_index += 1
        else:
            actual_index += 1
    return indices

c = u'\u2015'    
def process_aligns(als):
    aligns = []
    for align in als.split():
        try:
            if align.find(c) > -1:
                en = align.split(c)[0]
                hu = align.split(c)[1]
            else:
                en = align.split("-")[0]
                hu = align.split("-")[1]
        except IndexError:
            sys.stderr.write("%s\n" % als)
            raise Exception("Malformed aligned file")
        
        aligns.append((en, hu))
    return aligns
  
def validate_aligns(sen):
    en_sen = sen["en_sen"]
    hu_sen = sen["hu_sen"]
    aligns = sen["aligns"]
    
    en_np_indices = set([tok[0] for tok in en_sen if isinstance(tok, tuple)])
    hu_np_indices = set([tok[0] for tok in hu_sen if isinstance(tok, tuple)])
    for align in aligns:
        en_i = int(align[0].strip("sb"))
        hu_i = int(align[1].strip("sb"))
        if  not ( ( en_i in en_np_indices ) and ( hu_i in hu_np_indices ) ) :
            sys.stderr.write("%d-%d\n" % (en_i, hu_i) )
            raise InvalidAlignException


def process_baseline(path):
    empty_sentence = {
    'id': None,
    'en_sen': None,
    'hu_sen': None,
    'sentence_hun': None,
    'aligns': None
    }
    sentences = []

    actual_sentence = dict(empty_sentence)
    state = "empty"
    
    with open(path) as runga_input_file:
        for line in runga_input_file:
            if state == "empty":
                try:
                    actual_sentence["id"] = int(line.strip())
                except ValueError:
                    # reached end of file or malformed input
                    continue
                state = "got_id"
                continue
            elif state == "got_id":
                try:
                    actual_sentence["en_sen"] = process_sen(line.strip())
                except InvalidBracketsException:
                    raise InvalidBracketsException("Invalid English bracketing in sentence: %d\n" % actual_sentence["id"])
                except InvalidNumberingException:
                    raise InvalidNumberingException("Invalid English np numbering in sentence: %d\n" % actual_sentence["id"])
                except:
                    raise Exception("Unknown error in english sentence: %d\n" % actual_sentence["id"])
                state = "got_en"
            elif state == "got_en":
                try:
                    assert(line.strip() == "")
                except AssertionError:
                    print(line)
                    raise Exception("MyOwn")
                state = "wait_for_hu"
                continue
            elif state == "wait_for_hu":
                try:
                    actual_sentence["hu_sen"] = process_sen(line.strip())
                except InvalidBracketsException:
                    raise InvalidBracketsException("Invalid Hungarian bracketing in sentence: %d\n" % actual_sentence["id"])
                except InvalidNumberingException:
                    raise InvalidNumberingException("Invalid Hungarian np numbering in sentence: %d\n" % actual_sentence["id"])
                state = "got_hu"
                continue
            elif state == "got_hu":
                assert(line.strip() == "")
                state = "wait_for_aligns"
                continue
            elif state == "wait_for_aligns":
                actual_sentence["aligns"] = process_aligns(line.strip())
                try:
                    validate_aligns(actual_sentence)
                except InvalidAlignException:
                    raise InvalidAlignException("Invalid np alignment in sentence: %d\n" % actual_sentence["id"])
                state = "got_align"
                continue
            elif state == "got_align":
                try:
                    assert(line.strip() == "")
                except AssertionError:
                    sys.stderr.write("%d\n" % actual_sentence["id"])
                    raise Exception("Malformed input file")
                state = "wait_for_last_but_on_line"
                continue
            elif state == "wait_for_last_but_on_line":
                assert(line.strip() == "")
                state = "wait_for_last_line"
                continue
            elif state == "wait_for_last_line":
                if not line.strip() == "":
                    sys.stderr.write("Missing last empty line after sentence: %d\n" % actual_sentence["id"])
                    continue
                sentences.append(actual_sentence)
                actual_sentence = dict(empty_sentence)
                state = "empty"
                continue
    return sentences