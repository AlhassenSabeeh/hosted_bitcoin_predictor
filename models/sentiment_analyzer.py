import mwclient
import time
import pandas as pd
import numpy as np
from transformers import pipeline
from statistics import mean
import warnings
warnings.filterwarnings("ignore")

class WikipediaSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
    
    def fetch_wikipedia_data(self):
        """Step 1: Fetch Wikipedia revisions and analyze sentiment"""
        print("Fetching Wikipedia data...")
        
        site = mwclient.Site("en.wikipedia.org")
        site.rate_limit_wait = True
        site.rate_limit_grace = 60
        page = site.pages["Bitcoin"]

        revs = []
        continue_param = None
        start_date = '2010-01-01T00:00:00Z'

        while True:
            params = {
                'action': 'query', 
                'prop': 'revisions', 
                'titles': page.name, 
                'rvdir': 'newer', 
                'rvprop': 'ids|timestamp|flags|comment|user', 
                'rvlimit': 500, 
                'rvstart': start_date
            }
            if continue_param:
                params.update(continue_param)

            response = site.api(**params)

            for page_id in response['query']['pages']:
                if 'revisions' in response['query']['pages'][page_id]:
                    revs.extend(response['query']['pages'][page_id]['revisions'])

            if 'continue' in response:
                continue_param = response['continue']
                time.sleep(2)
            else:
                break

        return revs

    def analyze_sentiment(self, revs):
        """Analyze sentiment of Wikipedia revisions"""
        print("Analyzing sentiment...")
        
        revs_df = pd.DataFrame(revs)
        
        def find_sentiment(text):
            if not text or str(text) == 'nan':
                return 0
            try:
                sent = self.sentiment_pipeline([str(text)[:250]])[0]
                score = sent["score"]
                if sent["label"] == "NEGATIVE":
                    score *= -1
                return score
            except:
                return 0

        edits = {}
        for index, row in revs_df.iterrows():
            date = time.strftime("%Y-%m-%d", time.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%SZ"))
            if date not in edits:
                edits[date] = dict(sentiments=list(), edit_count=0)

            edits[date]["edit_count"] += 1
            comment = row.get("comment", "")
            if isinstance(comment, float) and np.isnan(comment):
                comment = ""
            edits[date]["sentiments"].append(find_sentiment(comment))

        # Aggregate by date
        for key in edits:
            if len(edits[key]["sentiments"]) > 0:
                edits[key]["sentiment"] = mean(edits[key]["sentiments"])
                edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
            else:
                edits[key]["sentiment"] = 0
                edits[key]["neg_sentiment"] = 0
            del edits[key]["sentiments"]

        edits_df = pd.DataFrame.from_dict(edits, orient="index")
        edits_df.index = pd.to_datetime(edits_df.index)
        
        return edits_df

    def create_sentiment_file(self):
        """Main function to create sentiment CSV file"""
        revs = self.fetch_wikipedia_data()
        edits_df = self.analyze_sentiment(revs)
        
        # Fill missing dates and apply rolling average
        from datetime import datetime
        dates = pd.date_range(start="2010-03-08", end=datetime.today())
        edits_df = edits_df.reindex(dates, fill_value=0)
        
        rolling_edits = edits_df.rolling(30, min_periods=30).mean()
        rolling_edits = rolling_edits.dropna()
        
        rolling_edits.to_csv("wikipedia_edits.csv")
        print("Sentiment analysis complete. File saved as 'wikipedia_edits.csv'")
        
        return rolling_edits
    