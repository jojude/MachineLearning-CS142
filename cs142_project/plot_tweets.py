#
#  plot_tweets.py
#  plot_tweets
#
#  Created by Jude Joseph on 03/23/17.

import matplotlib.pyplot as plt
import csv

# initial arrays to store values
dates = []
sentiment_value = [0]

# DEF: GET_RESULTS
# read csv file , iterate through
# add up polarity for each tweet
# on x day. Put that value into
# sentiment_value matched with its
# date.
def get_results(file):
	csv_f = csv.reader(file)
	polarity= 0.0
	date = ''

	for row in csv_f:
		date = row[0]
		polarity = polarity + float(row[2])

	dates.append(date)
	sentiment_value.append(polarity)

#open files pertaining to dates MAR-19 to MAR-23
#19
f = open('result.csv')
get_results(f)
#20
f2 =open('result2.csv')
get_results(f2)
#21
f3 =open('result3.csv')
get_results(f3)
#22
f4 =open('result4.csv')
get_results(f4)
#23
f5 =open('result5.csv')
get_results(f5)

print(dates)
print(sentiment_value)

# print out plot of change in
# sentiments toward 'Google'
# that we calculated through
fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,6], sentiment_value)
fig.suptitle('Twitter: Google Tweets Sentiment Analysis', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Positive Polarity')
ax.set_xlabel('2017-Mar-19 to 2017-Mar-23')
plt.show()

#end
