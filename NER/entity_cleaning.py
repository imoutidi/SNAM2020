import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Person entities that consist of only one word
# are being deleted to reduce ambiguity
def remove_single_named_persons(name_list):
    new_name_list = []
    for entity in name_list:
        if len(entity.strip().split(" ")) > 1:
            new_name_list.append(entity)
    return new_name_list


# Using the annotations dictionary to further
# remove/clean entities.
def second_phase_name_cleaning(name_list, clean_dict):
    new_name_list = []
    for entity in name_list:
        if entity in clean_dict:
            new_name_list.append(clean_dict[entity])
        else:
            new_name_list.append(entity)
    return new_name_list


# Removing stop words from each string of the given list
def remove_stop_words(list_of_strings):
    stop_words = set(stopwords.words("english"))
    for idx, entity in enumerate(list_of_strings):
        words = word_tokenize(entity)
        filtered_sentence = []
        for w in words:
            if w not in stop_words:
                filtered_sentence.append(w)
        string_entity = ""
        for w in filtered_sentence:
            string_entity += w + " "
        string_entity = string_entity[:-1]
        list_of_strings[idx] = string_entity


def person_cleaning(ent_list, clean_persons):
    ent_list = remove_single_named_persons(ent_list)
    # clean name dictionary
    ent_list = second_phase_name_cleaning(ent_list, clean_persons)
    remove_stop_words(ent_list)

    return ent_list

# the clean dictionaries are created by using the most frequent entities detected on previous runs
def clean(article_list):
    project_path = "/home/iraklis/PycharmProjects/AllTheNews/"
    load_clean_persons = open(project_path + "Pivot_Files/Clean_Entity_Dictionaries/clean_persons_dict.pickle", "rb")
    clean_persons_dict = pickle.load(load_clean_persons)
    load_clean_persons.close()

    load_clean_locs = open(project_path + "Pivot_Files/Clean_Entity_Dictionaries/clean_loc_dict.pickle", "rb")
    clean_locs = pickle.load(load_clean_locs)
    load_clean_locs.close()

    load_clean_orgs = open(project_path + "Pivot_Files/Clean_Entity_Dictionaries/clean_org_dict.pickle", "rb")
    clean_orgs = pickle.load(load_clean_orgs)
    load_clean_orgs.close()


    for sent_list in article_list:
        for sentence_entities in sent_list:
            sentence_entities["P"] = person_cleaning(sentence_entities["P"], clean_persons_dict)
            sentence_entities["L"] = second_phase_name_cleaning(sentence_entities["L"], clean_locs)
            sentence_entities["O"] = second_phase_name_cleaning(sentence_entities["O"], clean_orgs)

    return article_list
