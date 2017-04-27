#
#  import_tweepy.py
#  import_tweepy
#
#  Created by Jude Joseph on 03/23/17.
#
#  Note : change 'q=' to query different things
#         change  'since' and 'until' to find tweets b/w different days
#         change 'result.csv' to write results in a different file
#

import tweepy
import csv
import numpy as np
from textblob import TextBlob
np.random.seed(7)

#Use Our keys
consumer_key= 'b53JhZGxJ7oPWlILZBm1pNrjH'
consumer_secret= 'XnZ8ktUvOFPfQ41hJH8VmUJnti4mLM7ly0WFQPzc9gEgW9kxia'
access_token='4327489995-mD4Mq6LuAZZtW390bvxQE7dKUSmXtSSMBPxOPOL'
access_token_secret='8KEFFkMJgExUU30z1JkBmUwBQQX9gBdvvNN5zkgoI7aKe'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

public_tweets = [tweet for tweet in tweepy.Cursor(
                    api.search, q="Googe",lang='en',since='2017-03-22', until='2017-03-23').items(100)]
    

#Defining a threshold for each sentiment
threshold=0
pos_sent_tweet=0
neg_sent_tweet=0

#open csv file
csvFile = open('result.csv','a')
csvWriter = csv.writer(csvFile)

# go through each tweet and do sentiment analysis via textblob
# add values with respect to neg or pos polarity
# return overall result
for tweet in public_tweets:
    analysis=TextBlob(tweet.text)
    sentiment = analysis.sentiment.polarity
    if sentiment>=threshold:
        pos_sent_tweet=pos_sent_tweet+1
        csvWriter.writerow([tweet.created_at, tweet.text, sentiment])
    else:
        neg_sent_tweet=neg_sent_tweet+1
        csvWriter.writerow([tweet.created_at, tweet.text, sentiment])

if pos_sent_tweet>neg_sent_tweet:
    print ("Overall Positive")
    print (pos_sent_tweet)
    print (neg_sent_tweet)
else:
    print ("Overall Negative")
    print (pos_sent_tweet)
    print (neg_sent_tweet)

#end
