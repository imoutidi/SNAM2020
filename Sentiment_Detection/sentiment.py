from textblob import TextBlob
from ToolPack import tools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from collections import defaultdict

class ElectionSentiment:
    def __init__(self):
        # The keywords are prodused from the summarization of the detected event in the last step of
        #  Moutidis, I., Williams, H.T.: Utilizing complex networks for event detection in heterogeneous
        #  high-volume newsstreams. In: International Conference on Complex Networks and Their Applications,
        self.top_keywords = {"Labour", "labour", "Brexit", "brexit", "Tory", "tory", "Corbyn", "corbyn", "Scotland",
                             "scotland", "@BorisJohnson", "@jeremycorbyn", "Tories", "tories", "Boris", "boris",
                             "#GE2019", "@UKLabour", "NHS", "#GeneralElection2019", "SNP", "@Conservatives",
                             "conservative", "democracy", "England", "elections", "vote", "votes", "voting"}
        self.io_path = "PATH FOR I/O FILES"
        self.election_tweets = list()

    def filter_election_tweets(self):
        all_tweets = tools.load_pickle(self.io_path + "tweet_list")
        for tweet_tuple in all_tweets:
            tweet_words = tweet_tuple[0].split()
            if len(set(tweet_words).intersection(self.top_keywords)) > 0:
                self.election_tweets.append(tweet_tuple[0])

    def sentiment_textblob(self):
        """
        polarity is a float within the range [-1.0, 1.0]
        subjectivity is a float within the range [0.0, 1.0]                                                                           where 0.0 is very objective and 1.0 is very subjective.
        """
        tweet_polarities = list()
        for tweet in self.election_tweets:
            polarity = TextBlob(tweet).sentiment.polarity
            tweet_polarities.append(polarity)

        """
        in plt.hist() when density=True it normalizes the values so that the total AREA of the bars sums up to 1
        with the following line and with density=False the sum of all height sums up to 1 so we have the percentage
        of the population with the co responding x-axis value
        """
        weights = np.ones_like(tweet_polarities) / float(len(tweet_polarities))
        fig = plt.figure()
        plt.hist(tweet_polarities, 40, density=False, histtype='bar',
                 facecolor='b', alpha=0.5, weights=weights)
        plt.title("Polarity frequencies of Boris Johnson election win.")
        fig.savefig(self.io_path + "Figures/election_polarity")
        plt.show()

    def sentiment_nltk(self):
        """
        The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to
        the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
        This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
        Calling it a 'normalized, weighted composite score' is accurate.

        It is also useful for researchers who would like to set standardized thresholds for classifying sentences
        as either positive, neutral, or negative.
        Typical threshold values (used in the literature cited on this page) are:

            *positive sentiment: compound score >= 0.05
            *neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
            *negative sentiment: compound score <= -0.05

        The pos, neu, and neg scores are ratios for proportions of text that fall in each category
        (so these should all add up to be 1... or close to it with float operation).
        These are the most useful metrics if you want multidimensional measures of sentiment for a given sentence.
        """

        sid = SentimentIntensityAnalyzer()
        tweet_polarities = list()
        polarity_class = defaultdict(int)
        for tweet in self.election_tweets:
            scores = sid.polarity_scores(tweet)
            if scores["compound"] >= 0.05:
                polarity_class["positive"] += 1
            if scores["compound"] <= -0.05:
                polarity_class["negative"] += 1
            if scores["compound"] < 0.05 or scores["compound"] > -0.05:
                polarity_class["neutral"] += 1
            if scores["compound"] >= 0.05 or scores["compound"] <= -0.05:
                tweet_polarities.append(scores["compound"])

        weights = np.ones_like(tweet_polarities) * 100 / float(len(tweet_polarities))
        fig = plt.figure()
        plt.hist(tweet_polarities, 40, density=False, histtype='bar',
                 facecolor='b', alpha=0.5, weights=weights)
        plt.title("Sentiment polarity frequencies of Boris Johnson's election.")
        plt.xlabel('Polarity metric')
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.ylabel('Percentage of tweets')
        fig.savefig("PATH FOR FIGURE EXTRACTION")
        for key, value in polarity_class.items():
            print(key, value)
        # plt.show()


if __name__ == "__main__":
    el_sentiment = ElectionSentiment()
    el_sentiment.filter_election_tweets()
    # el_sentiment.sentiment_textblob()
    el_sentiment.sentiment_nltk()
