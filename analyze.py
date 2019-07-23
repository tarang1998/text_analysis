from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

import sentiment_mod as s


#consumer key, consumer secret, access token, access secret.
ckey="IpZAQKfjIUrUWwN13uqLVloQv"
csecret="5nFWnUjbXSfoWpilUUCBbHbXSN0LHdRfemvhBbeneBuOsjScoc"
atoken="1147083534378704896-ePOZ1bn0l6ec9IBPZ3ARNnADGNdNCH "
asecret="rsaaHbuaiONQGk74qRfjD2EKIV1d112dwnhB9GOLQwK37 "

class listener(StreamListener):
    
    def on_data(self, data):
        
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value,confidence=s.sentiment(tweet)  
        print(tweet,sentiment_value, confidence)

        if confidence>=80:
            output=open("twitter-out.txt",'a')
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        
        return True

    def on_error(self, status):
        print(status)


        

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)



twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])


'''
print(s.sentiment("This movie was awesome, superb, beautiful, legendary, full of suspense, a must watch"))
print(s.sentiment("This movie was utter junk, unimaginative, a complete peice of shit, must avoid it "))
'''
