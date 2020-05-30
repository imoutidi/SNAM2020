from nltk import word_tokenize
from nltk.tag import StanfordNERTagger
from collections import OrderedDict
from NER import entity_tools



# Removing entities that are part of bigger ones ("John", "John Oliver" we remove John)
# The entitiesList was a set() that were converted to a list
def remove_entities(entities_list):
    new_entities_list = []
    counter = 0
    for ent in entities_list:
        for secEnt in entities_list:
            if ent in secEnt:
                counter += 1
        if counter == 1:
            ent = remove_stop_words(ent)
            new_entities_list.append(ent)
        counter = 0
    return new_entities_list


def remove_stop_words(entity):
    new_entity = ""
    custom_stop_words = ["Mr.", "Mr", "Mister", "Dr.", "Dr", "Prof.", "Prof", "Miss", "Mrs.",
                         "Mrs", "Ms.", "Ms", "Master", "Sr.", "Sr", "Sr.", "Br.",
                         "Br", "Fr.", "Fr", "Esq", "Mx.", "Mx", "Jr", "Jr.", "J", "J.", "Sgt"]

    custom_stop_characters = ["--", " [", "[ ", "[", " ]", "] ", "]", "– ", " –", "–", "— ", " —", "—",
                              " ’", "’ ", "’", " ”", "” ", "”", " “", "“ ", "“", " (", "( ", "(", " )",
                              ") ", ")", " J.", " ,", ", ", ",", "*"]

    for ent_word in entity.split(" "):
        if ent_word not in custom_stop_words:
            for sc in custom_stop_characters:
                if sc in ent_word:
                    ent_word = ent_word.replace(sc, '')
            new_entity += ent_word + " "

    return new_entity.strip()


def detection(document):
    # document type:str
    # stanford's NER tagger 3 entity classification PERSON LOCATION ORGANIZATION O(other)
    # Those file must be dowloaded manually.
    stan_ner = StanfordNERTagger('/home/iraklis/Desktop/PhDLocal/Tools/stanford-ner-2017-06-09/classifiers/'
                                 'english.all.3class.distsim.crf.ser.gz',
                                 '/home/iraklis/Desktop/PhDLocal/Tools/stanford-ner-2017-06-09/'
                                 'stanford-ner.jar',
                                 encoding='utf-8')
    tokenized_text = word_tokenize(document)
    classified_text = stan_ner.tag(tokenized_text)
    # Dirty code shame on me.
    if len(classified_text) == 0:
        return "nan"

    # write_on_file(classified_text)
    # Chunking entities (for example first name with last name)
    previous_tuple = classified_text[0]
    entity_str = ""
    if previous_tuple[1] != 'O':
        entity_str += previous_tuple[0]

    store_entities = dict()
    store_entities["P"] = []
    store_entities["L"] = []
    store_entities["O"] = []

    for txt in classified_text[1:-1]:
        if txt[1] != 'O':
            if txt[1] == previous_tuple[1]:
                entity_str += " " + txt[0]
            else:
                entity_str = txt[0]
        else:
            if entity_str != "":
                store_entities[previous_tuple[1][0]].append(entity_str)
                entity_str = ""

        previous_tuple = txt

    # We are using OrderedDict to delete duplicates and preserve the insertion order
    store_entities["P"] = remove_entities(
        list(OrderedDict((x, True) for x in store_entities["P"]).keys()))
    store_entities["L"] = remove_entities(
        list(OrderedDict((x, True) for x in store_entities["L"]).keys()))
    store_entities["O"] = remove_entities(
        list(OrderedDict((x, True) for x in store_entities["O"]).keys()))

    # We transform the article_entity_dict_list in order to contain the entities of
    # each article sentence. The article entities will be calculated when it is needed.
    sentence_entities = entity_tools.transform_article_dict(document, store_entities)

    return sentence_entities

