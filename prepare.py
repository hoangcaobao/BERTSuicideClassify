import pandas as pd

def csv_to_data(link, text_title, label_title, label_category):
    #get data and label from csv file
    df=pd.read_csv(link)
    texts=list(df[text_title])
    labels=list(df[label_title])
    for i in range(len(labels)):
        if(labels[i]==label_category):
            labels[i]=1
        else:
            labels[i]=0
    return texts, labels