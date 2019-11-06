import json
import spacy
import pandas as pd
from train import train_spacy,get_scores


def getEnts(data):
    # print(data)
    entity = data[0]
    message = data[1]
    
    ents = []
    for i in entity:
        
        if 'entityName' in i.keys():
            ent = []
            ent.append(i["start"])
            ent.append(i["end"])
            ent.append(i["entityName"])
            ents.append(ent)
        # else:
            # print(i)
    j = {}
    j["entities"] = ents
    
    tup = []
    tup.append(message)
    tup.append(j)
    
    
    return tuple(tup)



if __name__ == "__main__":
    
    connection_file = open("./data/data.json", 'r')
    data = json.load(connection_file)

    df = pd.DataFrame(data)

    dropCol = ['path', 'subCategory',  'vertical',  'category','card', 'mask']


    df.drop(columns = dropCol,inplace=True);


    df = df[df['entities'].map(lambda d: len(d)) > 0]


    df["train"] = df.apply( getEnts,axis=1)
    df.to_csv("./data/readyForTrain.csv",index=True)

    # [{"content":"abc is a boss","entities":[[0,3,"a"],[4,6,"b"],[0,1,"c"],[9,13,"d"]]}]


    prdnlp = train_spacy(df["train"].tolist(), 50)
    
    # Save our trained Model
    prdnlp.to_disk('nerModel')
    
    
    prdnlp = spacy.load("nerModel")

    score = get_scores(prdnlp, df["train"].tolist())
    print(score)
    while True:
        test_text = input("Enter your testing text: ")
        doc = prdnlp(test_text)
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

        