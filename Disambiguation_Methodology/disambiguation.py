import re
import json
import glob
import requests
from ToolPack import tools
from bs4 import BeautifulSoup
from NER import entity_detection
from nltk.corpus import stopwords
from nltk import edit_distance
import numpy as np
from Disambiguation_Methodology import kmedoids
from collections import defaultdict


class ElectionSentiment:
    def __init__(self):
        # dowload the Disambiguation folder from here https://drive.google.com/open?id=1SJLWSooKfXZsb3BBxWVvOjZN4za4yu8o
        # and adjust the path.
        self.io_path = "PATH TO Disambiguation FOLDER"
        # The Ritter classifier can be found here http://github.com/aritter/twitter_nlp
        self.ritter_classifier_path = "PATH FOR THE RITTER TWEET NER CLASSIFIER"
        self.retrieved_tweets = list()
        self.url_text_entities = list()
        self.tweets_entities = list()
        self.test_tweets = list()
        self.stop_words = stopwords.words("english")

    # gathering tweets from a given collection, detects any urls on them and saves their id, text and detected url.
    def process_tweets(self):
        # the election tweets file is too big to be pushed in github
        # it can be found here https://drive.google.com/open?id=1SJLWSooKfXZsb3BBxWVvOjZN4za4yu8o
        tweet_list = tools.load_pickle(self.io_path + "election_tweets")
        for tweet in tweet_list:
            urls = re.findall(r'http\S+', tweet.text)
            if len(urls) > 0:
                self.retrieved_tweets.append((tweet.id, tweet.text, urls[0]))

    # for NER detection on tweets we use the implementation of  Ritter, A., Clark, S., Mausam, Etzioni, O.
    # (Named entity recognition in tweets: An experimental study.) which can not provide with our code.
    def tweet_ne_detection(self):
        with open(self.ritter_classifier_path + "example_tweets", 'w') as output_tweets:
            with open(self.io_path + "url_tweets") as input_tweets:
                for tweet in input_tweets:
                    # removing the tweet id from the text
                    tweet_text = ' '.join(tweet.split()[1:])
                    output_tweets.write("tweet_text " + self.clean_string(tweet_text) + "\n")
        # we can not provide the Ritter classifier.
        # The reader can download it here: http://github.com/aritter/twitter_nlp
        # and run the classifier manually.

    # using the links (if they exist) of each tweet to scrape its full text.
    def retrieve_url_text(self):
        with open(self.io_path + "url_tweets", 'w') as url_tweets, open(self.io_path + "url_text", 'w') as url_text:
            for tweet_tuple in self.retrieved_tweets[2000:4000]:
                url_body = 'text '
                blacklist = ["document", "noscript", "header", "meta", "script", "head", "html",
                             "input", "style", "data-language"]
                try:
                    req_get = requests.get(tweet_tuple[2], timeout=15)
                    url_content = req_get.content
                    b_soup = BeautifulSoup(url_content, 'html.parser')
                    url_struct = b_soup.find_all(text=True)
                    for t in url_struct:
                        if t.parent.name not in blacklist:
                            url_body += '{} '.format(t)

                    cleaned_tweet_text = self.clean_string(" ".join(tweet_tuple[1].splitlines()))
                    url_tweets.write(str(tweet_tuple[0]) + ' ' + cleaned_tweet_text + "\n")

                    cleaned_body = self.clean_string(url_body.replace("\n", ""))
                    cleaned_body = " ".join(cleaned_body.splitlines())
                    url_text.write(cleaned_body + "\n")

                    print(cleaned_body)
                    print("-----")
                except requests.exceptions.ConnectionError as conn_error:
                    print(conn_error)
                    print(tweet_tuple[2])
                except requests.exceptions.MissingSchema as sc_error:
                    print(sc_error)
                    print(tweet_tuple[2])
                except requests.exceptions.InvalidURL as inv_error:
                    print(inv_error)
                    print(tweet_tuple[2])
                except requests.exceptions.ContentDecodingError as dec_error:
                    print(dec_error)
                    print(tweet_tuple[2])
                except requests.exceptions.ChunkedEncodingError as enc_error:
                    print(enc_error)
                except requests.exceptions.ReadTimeout as timeout_error:
                    print(timeout_error)
                except UnicodeDecodeError as unicode_error:
                    print(unicode_error)

    # For simplicity we run our method on a sample of the tweets.
    # Detecting entities on the text linked on tweets.
    # Here we use the Stanford NER classifier of  Finkel, J.R., Grenager, T., Manning, C.
    # (Incorporating non-local information into information extraction systems by gibbs sampling.)
    def url_ne_detection(self):
        with open(self.io_path + "url_text") as url_text:
            for idx, doc in enumerate(url_text):
                print(idx)
                detected_entities = entity_detection.detection(doc)
                persons_set = set()
                for entity_tuple in detected_entities:
                    # P stands for person
                    persons_set |= entity_tuple['P']
                self.url_text_entities.append(persons_set)
        tools.save_pickle(self.io_path + "url_text_entities", self.url_text_entities)

    # Getting the Person entities for each tweet (if any) from the output of the classifier.
    def parse_tweets_classifications(self):
        # isolating the person entities from the classifier output
        with open(self.io_path + "classified_tweets") as class_file:
            for tweet in class_file:
                entities_in_tweet = list()
                # entities appear on the list with the order they appear in the tweet text
                entities_occured = list()
                splitted_ents = tweet.split()[1:]
                for ent in splitted_ents:
                    if "-person" in ent:
                        entities_occured.append(ent)
                # merge B-I persons
                while len(entities_occured) > 0:
                    current_name = ""
                    if "B-person" in entities_occured[0]:
                        current_name += entities_occured[0].split('/')[0]
                        del entities_occured[0]
                        if len(entities_occured) > 0:
                            while "I-person" in entities_occured[0]:
                                current_name += ' ' + entities_occured[0].split('/')[0]
                                del entities_occured[0]
                                if len(entities_occured) == 0:
                                    break
                        entities_in_tweet.append(current_name)
                self.tweets_entities.append(entities_in_tweet)

    # Disambiguating single word named entities on tweets using detected named entities
    # on documents linked from each tweet.
    def disambiguate_entities_phase1(self):
        # gathering only the sample tweets
        self.get_test_tweets()
        # can be removed after testing
        self.url_text_entities = tools.load_pickle(self.io_path + "url_text_entities")
        with open(self.io_path + "phase1_disampiguated_entities", 'w') as dis_file:
            for tweet_ents, url_ents, tweet_text in zip(self.tweets_entities, self.url_text_entities, self.test_tweets):
                for idx, t_entity in enumerate(tweet_ents):
                    if len(t_entity.split()) == 1:
                        for url_entity in url_ents:
                            if t_entity in url_entity:
                                tweet_ents[idx] = url_entity
                                tweet_text = tweet_text.replace(t_entity, url_entity)
                                dis_file.write("Replaced " + t_entity + " with " + url_entity + "\n")

    # Creating clusters/super documents of tweets and running the second phase of
    # the disambiguation process
    def disambiguate_entities_phase2(self):
        # This calculation can take some time. Once it is run it can be commented
        self.edit_distance_calc()
        distances_array = tools.load_pickle(self.io_path + "distance_2dlist")
        medoid, clusters = kmedoids.kmedoids(distances_array, 3)

        # creating 'super documents' of named entities. Given that we have already detected and applied the
        # phase1 of the disambiguation process we will use the collections of named entities for each tweet
        # and not the tweet itself.
        clustered_entities = list()
        for c_id, clust in clusters.items():
            entity_clust = list()
            for tweet_id in clust:
                entity_clust.append(self.tweets_entities[tweet_id])
            clustered_entities.append(entity_clust)

        # creating entity ranking for each cluster
        list_of_rankings = list()
        for entity_cluster in clustered_entities:
            cluster_dict = defaultdict(int)
            for entity_list in entity_cluster:
                if len(entity_list) > 0:
                    for entity in entity_list:
                        if len(entity.split()) > 1:
                            cluster_dict[entity] += 1
            ranking = list()
            for entity, frequency in cluster_dict.items():
                ranking.append((entity, frequency))
            ranking.sort(key=lambda tup: tup[1], reverse=True)
            list_of_rankings.append(ranking)

        # replacing single name entities according to the cluster ranking that the tweet containing it is.
        with open(self.io_path + "phase2_disampiguated_entities", 'w') as dis2_file:
            for c_id, clust in clusters.items():
                for tweet_idx in clust:
                    for ent_idx, tweet_ent in enumerate(self.tweets_entities[tweet_idx]):
                        if len(tweet_ent.split()) == 1:
                            for ranked_entity in list_of_rankings[c_id]:
                                if tweet_ent in ranked_entity[0]:
                                    dis2_file.write("Replaced " + tweet_ent + " with " + ranked_entity[0] + "\n")
                                    self.tweets_entities[tweet_idx][ent_idx] = ranked_entity[0]
                                    break
        print()

    # --------------------------------------------------
    # utility functions
    # --------------------------------------------------
    def parse_json2(self):
        id_list = list()
        total = 0
        for ppath in glob.iglob("/media/iraklis/Elements/Data Backup/Tweeter_Graphs/US_Elections/NER_Tweet_Text/*"):
            with open(ppath) as f:
                for i, l in enumerate(f):
                    pass
                total += i
        print(total)
        print()
        for filepath in glob.iglob("/media/iraklis/Elements/Data Backup/Tweeter_Graphs/US_Elections/"
                                   "US_Election_Tweets/*.json"):
            print(filepath)
            with open(filepath, 'r') as tweet_json_file:
                for idx, tweet_line in enumerate(tweet_json_file):
                    try:
                        tweet_dict = json.loads(tweet_line.strip())
                        if "id" in tweet_dict:
                            id_list.append(tweet_dict["id"])
                    except json.decoder.JSONDecodeError as e:
                        print(e)
                        print("Error in line " + str(idx+1) + " of json file.")

        print(len(id_list))
        with open("/home/iraklis/Desktop/2012_tweets_ids", 'w') as id_file:
            for id in id_list:
                id_file.write(id[9:] + "\n")

    @staticmethod
    def clean_string(input_str):
        ascii_str = ""
        # removing all non ascii characters
        for letter in input_str:
            num = ord(letter)
            if 0 <= num <= 127:
                ascii_str += letter
        ascii_str = ascii_str.replace("\n", "")
        return ascii_str

    def get_test_tweets(self):
        with open(self.io_path + "url_tweets") as tweet_reader:
            for tweet_line in tweet_reader:
                splited_tweet = tweet_line.split()[1:]
                self.test_tweets.append(" ".join(splited_tweet))

    def remove_stop_words(self, tweet_string):
        tweet_string = re.sub(r'http\S+', '', tweet_string)
        tweet_string = ' '.join([word for word in tweet_string.split() if word not in self.stop_words])
        return tweet_string

    def edit_distance_calc(self):
        # Calculating the edit distances between all tweets
        distances_array = np.zeros(shape=(len(self.test_tweets), len(self.test_tweets)))
        for i in range(len(self.test_tweets)):
            print(i)
            for j in range(len(self.test_tweets)):
                # we need to calculate only the upper triangular matrix
                # but we need the matrix to be full
                if i <= j:
                    distances_array[i][j] = edit_distance(self.remove_stop_words(self.test_tweets[i]),
                                                          self.remove_stop_words(self.test_tweets[j]))
                else:
                    distances_array[i][j] = distances_array[j][i]
        tools.save_pickle(self.io_path + "distance_2dlist", distances_array)



if __name__ == "__main__":
    ambi_sent = ElectionSentiment()
    ambi_sent.process_tweets()
    ambi_sent.tweet_ne_detection()
    ambi_sent.retrieve_url_text()
    ambi_sent.url_ne_detection()
    ambi_sent.parse_tweets_classifications()
    ambi_sent.disambiguate_entities_phase1()
    ambi_sent.disambiguate_entities_phase2()

    print()