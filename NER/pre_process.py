import glob
import json
from datetime import datetime
from ToolPack import tools


def prep_for_ner(doc_path):
    # file_counter = 0
    file_idx_to_date = dict()
    date_to_timestamp = dict()
    for file_idx, file_path in enumerate(glob.iglob(doc_path + "US_Election_Tweets/*")):
        # counting tweets
        num_lines = sum(1 for line in open(file_path))
        with open(file_path, "r", encoding='ISO-8859-1') as json_file:
            json_line = json_file.readline()
            data = json.loads(json_line)
            stamp = int(data["publicationTime"]) / 1000
            exact_date = datetime.utcfromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S')
            file_idx_to_date[file_idx] = exact_date
            date_to_timestamp[exact_date] = stamp
            with open(doc_path + "NER_Tweet_Text/" + str(file_idx), "wb") as text_file:
                tweet = data["title"].replace("\n", "")
                tweet = tweet + "\n"
                tweet = tweet.encode("ascii", "ignore")
                text_file.write(tweet)

                # -1 because we already red the first line of the json doc
                count = 1
                for i in range(num_lines - 1):
                    json_line = json_file.readline()
                    data = json.loads(json_line)
                    tweet_string = data["title"].replace("\n", "")
                    tweet_string = tweet_string + "\n"
                    tweet_string = tweet_string.encode("ascii", "ignore")
                    if len(tweet_string) > 1:
                        text_file.write(tweet_string)
    tools.save_pickle(doc_path + "NER_Tweet_Text/idx_to_date.pickle", file_idx_to_date)
    tools.save_pickle(doc_path + "NER_Tweet_Text/date_to_timestamp.pickle", date_to_timestamp)


def get_date_order(dataset):
    pivot_path = "/home/iraklis/PycharmProjects/Tweeter_Graphs/I_O/Tweeter_Datasets/" + dataset + "/Pivot_Files/"
    date_order = tools.load_pickle(pivot_path + "date_order.pickle")
    date_to_stamp = tools.load_pickle("/home/iraklis/PycharmProjects/Tweeter_Graphs/I_O/Tweeter_Datasets/" + dataset +
                                      "/NER_Tweet_Text/date_to_timestamp.pickle")
    # create a stamp to date dictionary
    stamp_to_date = dict()
    for c_date, stamp in date_to_stamp.items():
        stamp_to_date[stamp] = c_date
    # relate the ids of the order file with actual dates
    order_with_date = list()
    for c_date in date_order:
        order_with_date.append((c_date[0], stamp_to_date[c_date[1]]))
    return order_with_date


if __name__ == "__main__":
    # d_path = "/home/iraklis/PycharmProjects/Tweeter_Graphs/I_O/Tweeter_Datasets/US_Elections/"
    # prep_for_ner(d_path)
    a = get_date_order("FA_Cup")
    print()

