import spacy
import random
from spacy.gold import GoldParse
from spacy.scorer import Scorer



def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                # print(text,"-->",annotations)
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.0,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                except Exception as e:
                    # print (e)
                    pass
            print(losses)
    return nlp


def get_scores(nlp, examples):
    """
    calculate the score of train data
    :param nlp:
    :param examples:
    :return: scorer.scores
    """
    loss = 0
    random.shuffle(examples)

    print("Iter", "Loss", "P", "R", "F")

    scores = evaluate(nlp, examples)
    report_scores(loss, scores)
    return scores


def evaluate(nlp, dev_sents):
    scorer = Scorer()
    for raw_text, annotations in dev_sents:
        doc = nlp.make_doc(str(raw_text))
        try:
            gold = GoldParse(doc, entities=annotations.get('entities'))
            # print(nlp.pipe_names,nlp.pipeline)
            # nlp.tagger(doc)
            nlp.entity(doc)
            # nlp.parser(doc)
            scorer.score(doc, gold)
        except Exception as e:
            pass
    return scorer.scores


def report_scores(loss, scores):
    """
    prints precision recall and f_measure
    :param scores:
    :param loss:
    :return:
    """
    precision = '%.2f' % scores['ents_p']
    recall = '%.2f' % scores['ents_r']
    f_measure = '%.2f' % scores['ents_f']
    print('%d %s %s %s' % (int(loss), precision, recall, f_measure))
    
    
    
    
    
if __name__ == "__main__":
    
    TRAIN_DATA = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}), ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]})]
    
    prdnlp = train_spacy(TRAIN_DATA, 10)
    
    # Save our trained Model
    modelfile = input("Enter your Model Name: ")
    # prdnlp.to_disk(modelfile)

    #Test your text
    test_text = input("Enter your testing text: ")
    doc = prdnlp(test_text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


    score = get_scores(prdnlp, TRAIN_DATA)
    print(score)
    
    
    pass
