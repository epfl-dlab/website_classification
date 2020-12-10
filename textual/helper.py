import re
from bs4 import BeautifulSoup



def bert_clean_sentence(s):
    """ Clean a sentence so that it is only composed of words """
    
    s = re.sub(r"’", "'", s) # transform the french ' into english ones
    
    match = r"[^a-zA-z\u00C0-\u00FF '’-]" # match anything that is not a letter (incl. accentued), space, apostrophe and dash 
    match += "|" + "(\W\-|\-\W)+" # match any dash that is not between letters
    match += "|" + "(\W'|'\W)+" # match any apostrophe that is not between letters
    
    s = re.sub(match, "", s) # remove the matches characters
    s = re.sub(r"\s+"," ", s) # replace any whitespace with a space
    s = re.sub(r" +"," ", s) # remove any sucession of spaces
    s = s.strip() # trim the final sentence
    
    return s

def bert_split(body):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """
    
    try:
        soup = BeautifulSoup(body, 'html.parser')
        
    except Exception:
        return None
    
    a = soup.get_text('[SEP]').split('[SEP]') # separate text elements with special separators, the, splits
    b = [s.split('.') for s in a if len(s) != 0] # split text elements in sentences
    flat_b = [bert_clean_sentence(s) for sublist in b for s in sublist] # clean the sentences, flatten everything
    cleaned_b = [s for s in flat_b if (len(s) != 0 and len(s) <=510)] # only keep sentences between 1 and 10 words
    
    return cleaned_b
