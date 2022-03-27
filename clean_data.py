import re
import contractions
from sklearn.model_selection import train_test_split

def remove_punc(text):
  #remove all punctuation
  clean_text = re.sub(r'[^\w\s]', ' ', text)
  return clean_text

def remove_html(text):
  #remove all html tag
  cleaner=re.compile("<[^>]*>")
  clean_text=re.sub(cleaner, ' ', text)
  return clean_text

def remove_urls(text):
  #remove all link 
  clean_text=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
  return clean_text

def remove_emoji(text):
  #remove all emoji tag
  cleaner= re.compile("["
                      u"\U0001F600-\U0001F64F"  
                      u"\U0001F300-\U0001F5FF"
                      u"\U0001F680-\U0001F6FF"  
                      u"\U0001F1E0-\U0001F1FF"  
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      "]+", flags=re.UNICODE)
  clean_text=re.sub(cleaner, ' ', text)
  return clean_text

def remove_email(text):
  #remove email
  clean_text=re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', ' ', text)
  return clean_text

def remove_multi_space(text):
  #remove multiple space between words
  clean_text=re.sub(' +', ' ', text)
  return clean_text

def clean(texts):
  #apply all regex + some traditional method
  for i in range(len(texts)):
    texts[i]=remove_email(texts[i])
    texts[i]=remove_html(texts[i])
    texts[i]=remove_urls(texts[i])
    texts[i]=remove_emoji(texts[i])
    try:
      texts[i]=contractions.fix(texts[i])  
    except:
      pass
    texts[i]=remove_punc(texts[i])
    texts[i]=texts[i].replace('\n', ' ')
    texts[i]=texts[i].strip()
    texts[i]=texts[i].lower()
    texts[i]=remove_multi_space(texts[i])

def split(x, y, test_size):
  #split data
  x_train, x_valid, y_train, y_valid=train_test_split(x, y, test_size=test_size, random_state=42)
  return x_train, x_valid, y_train, y_valid
