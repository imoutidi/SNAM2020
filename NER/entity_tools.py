import copy
import pickle
from NER import entity_cleaning


def transform_article_dict(doc, ent_dict):
    project_path = "/home/iraklis/PycharmProjects/AllTheNews/"
    # Loading the sentence classifier
    load_classifier = open(project_path + "Pivot_Files/Classifiers/PunktSentenceTokenizer.pickle", "rb")
    p_sent_tokenizer = pickle.load(load_classifier)
    load_classifier.close()

    sentences = p_sent_tokenizer.sentences_from_text(doc)
    ent_dict["P"] = entity_cleaning.remove_single_named_persons(ent_dict["P"])
    list_of_sets = replace_entities_in_sentence(sentences, ent_dict)

    return list_of_sets


# In order to match each entity of the paragraph we have
# to replace all words that refer to the same entity with
# one unique word/s for example Theresa and May will be
# Theresa May and Theresa May we will later use the set
# container to erase duplicates per sentence.
def replace_entities_in_sentence(doc_sents, ent_list_doc):
    entities_in_sentence = dict()
    entities_in_sentence["P"] = set()
    entities_in_sentence["L"] = set()
    entities_in_sentence["O"] = set()

    new_sents = []
    set_list = []
    entity_time_list = []
    for single_sentence in doc_sents:
        correct_entities_per = replace_per(ent_list_doc["P"], single_sentence,
                                           entity_time_list)
        correct_entities_loc = replace_lo(ent_list_doc["L"], single_sentence)
        correct_entities_org = replace_lo(ent_list_doc["O"], single_sentence)

        for person in correct_entities_per:
            entities_in_sentence["P"].add(person)
        for location in correct_entities_loc:
            entities_in_sentence["L"].add(location)
        for organization in correct_entities_org:
            entities_in_sentence["O"].add(organization)

        # We need to deep copy the dictionary because the sets will be cleared.
        new_sents.append(copy.deepcopy(entities_in_sentence))
        entities_in_sentence["P"].clear()
        entities_in_sentence["L"].clear()
        entities_in_sentence["O"].clear()

    return new_sents


def replace_per(ent_list, sentence, time_list):
    temp_sent_ent = dict()
    whole_entities = set()
    for entity in ent_list:
        if entity in sentence:
            sentence = sentence.replace(entity, '')
            time_list.append(entity)
            temp_sent_ent[entity.split(" ")[0]] = [entity]
            whole_entities.add(entity)

    for wrd in sentence.split(" "):
        for ent in ent_list:
            striped_word = wrd.strip('\' ,.“[]()”—’:;?')
            for entSp in ent.split(" "):
                if striped_word == entSp:
                    if striped_word in temp_sent_ent:
                        temp_sent_ent[striped_word].append(ent.strip())
                    else:
                        temp_sent_ent[striped_word] = [ent.strip()]

    correct_entities = remove_wrong_entities(temp_sent_ent, time_list)
    all_entities = correct_entities.union(whole_entities)

    return all_entities


# If two and more entities match with a word (last/first name) we chose which
# entity we will keep by detecting the closest whole entity that were
# replaced in the sentence.
def remove_wrong_entities(sentence_entities, time_list):
    correct_entities = set()
    for key, value in sentence_entities.items():
        if len(value) > 1:
            if value[0] in time_list:
                max_candidate = [value[0], time_list.index(value[0])]
            else:
                max_candidate = [value[0], -1]
            for candidate_entity in value[1:]:
                if candidate_entity in time_list:
                    candidate_score = time_list.index(candidate_entity)
                else:
                    candidate_score = -1
                if max_candidate[1] < candidate_score:
                    max_candidate = [candidate_entity, candidate_score]
            if max_candidate[1] > -1:
                correct_entities.add(max_candidate[0])
        else:
            correct_entities.add(value[0])

    return correct_entities


def replace_lo(ent_list, sentence):
    temp_sent_ent = dict()
    sentence_entities = []
    for entity in ent_list:
        if entity in sentence:
            # deleting the word from the sentence so it wont de detected again
            sentence = sentence.replace(entity, '')
            sentence_entities.append(entity)
            temp_sent_ent[entity.split(" ")[0]] = [entity]

    return set(sentence_entities)