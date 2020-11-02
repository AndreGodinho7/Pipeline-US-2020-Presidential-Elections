######## EXCEPTIONS CONFIGS ########
from Exceptions import InvalidTweet, NotEnglishTweet




######## UTILS CONFIGS ########
import json
import time
import sys
import os
import gc
sys.path.append("..") # to be able to import modules


######## PREPROCESS CONFIGS ########
import re
from datetime import datetime
import emoji




######## LOGGING CONFIGS ########
import logging
LOGGING_FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'




######## KAFKA CONFIGS ########
from confluent_kafka import Consumer, KafkaError


## CONSUMER_CONFIG_PATH = '/home/andregodinho06/Projects/Twitter Project/consumer.json'
AUTO_OFFSET_RESET = 'earliest'
ENABLE_AUTO_COMMIT = False

## maximum number of messages to return
# to avoid issues consumers are encouraged to
# process data fast and poll often
MAX_POLL_RECORDS = 250

## maximum time to block waiting for message (in seconds)
MAX_BLOCK_WAIT_TIME = 2

## maximum amount of time between two .poll() calls before declaring the consumer dead
POLL_INTERVAL_MIN = 60
MAX_POLL_INTERVAL_MS = POLL_INTERVAL_MIN*60*1000 # TODO: how many milliseconds to wait between poll?



######## MODEL CONFIGS ########
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DistilBertTokenizer
from BERTSentimentAnalysis.DataPreProcess.BERTFormatDataloader import BERTFormatDataloader
from BERTSentimentAnalysis.DataPreProcess.TweetsDataset import TweetsDataset
# from BERTSentimentAnalysis.SentimentClassifier.BERTSentimentClassifier import BERTSentimentClassifier
# from BERTSentimentAnalysis.SentimentClassifier.BERT import BERT
from BERTSentimentAnalysis.SentimentClassifier.newBERTSentimentClassifier import BERTSentimentClassifier
from BERTSentimentAnalysis.SentimentClassifier.BERT import BERT
from BERTSentimentAnalysis.SentimentClassifier.DistillBERTSentimentClassifier import DistillBERTSentimentClassifier
from BERTSentimentAnalysis.SentimentClassifier.DistillBERT import DistillBERT


RANDOM_SEED = 10
MAX_LEN = 512
BATCH_SIZE = 16
n_GPU = 0
CLASSES = 2
class_names = ['negative', 'positive']
USE_GPU = False
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
#MODEL_NAME = '/home/andregodinho06/Projects/Twitter Project/bert_classifier.bin'
#MODEL_NAME = '/home/andregodinho06/Projects/Twitter Project/distillbert_classifier.bin'

######## ELASTICSEARCH CONFIGS ########
from elasticsearch import Elasticsearch
from elasticsearch import helpers


######## WORKER CONFIGS ########
import threading
from multiprocessing import Process, Barrier
from queue import Queue

NUM_WORKERS = 60
NUM_THREADS = 1
barrier = Barrier(NUM_WORKERS)


def create_kafka_config(jsonData):
    conf = {
        'bootstrap.servers': jsonData.get('bootstrapservers'),
        'group.id': jsonData.get('groupid'),
        'auto.offset.reset': AUTO_OFFSET_RESET, # 'earliest'
        'enable.auto.commit': ENABLE_AUTO_COMMIT, # False
        'max.poll.interval.ms' : MAX_POLL_INTERVAL_MS #TODO: tune (now it's 20 min)
    }

    return conf

def extract_tweet_info(record):
    record_json = json.loads(record)
    record_json_keys = list(record_json.keys())

    if 'id_str' not in record_json_keys:
        raise InvalidTweet

    if record_json['lang'] != "en":
        raise NotEnglishTweet

    tweet_id = record_json['id_str'] 
    # date example Tue Oct 20 18:54:06 +0000 2020
    tweet_date = datetime.strptime(record_json['created_at'], '%a %b %d %H:%M:%S %z %Y').isoformat()
    user_id = record_json['user']['id_str']
    user_name = record_json['user']['screen_name']
    user_location = record_json['user']['location']
    user_followers = record_json['user']['followers_count']
    user_friends = record_json['user']['friends_count']
    user_verified = record_json['user']['verified']
    user_created_at = datetime.strptime(record_json['user']['created_at'], '%a %b %d %H:%M:%S %z %Y').isoformat()

    # for key, value in record_json.items():
    #     print("key: %s | value: %s" %(key, value))

    if record_json['is_quote_status']: # quotes are usually against the user tweet message, do not use neither if it has a retweet
        if 'extended_tweet' in record_json_keys:
            text = record_json['extended_tweet']['full_text']
        else:
            text = record_json['text']
    else:
        if 'retweeted_status' in record_json_keys: # users usually agree with retweet opinions, forward it as tweet message
            retweet_json_keys = list(record_json['retweeted_status'].keys())

            if 'extended_tweet' in retweet_json_keys:
                text = record_json['retweeted_status']['extended_tweet']['full_text']
            else:
                text = record_json['retweeted_status']['text']

            # date example Tue Oct 20 18:54:06 +0000 2020
            ret_tweet_date = datetime.strptime(record_json['retweeted_status']['created_at'], '%a %b %d %H:%M:%S %z %Y').isoformat()
            ret_user_id = record_json['retweeted_status']['user']['id_str']
            ret_user_name = record_json['retweeted_status']['user']['screen_name']
            ret_user_location = record_json['retweeted_status']['user']['location']
            ret_user_followers = record_json['retweeted_status']['user']['followers_count']
            ret_user_friends = record_json['retweeted_status']['user']['friends_count']
            ret_user_verified = record_json['retweeted_status']['user']['verified']
            ret_user_created_at = datetime.strptime(record_json['retweeted_status']['user']['created_at'], '%a %b %d %H:%M:%S %z %Y').isoformat()

            return {
                tweet_id: {
                    "tweet": text,
                    "sentiment": '',
                    "tweet_created_at": tweet_date,
                    "user_id": user_id,
                    "user_name": user_name,
                    "user_location": user_location,
                    "user_followers": user_followers,
                    "user_friends": user_friends,
                    "user_verified": user_verified,
                    "user_created_at": user_created_at,
                    "retweet_user_name": ret_user_name, 
                    "retweet_user_location": ret_user_location, 
                    "retweet_user_followers": ret_user_followers,
                    "retweet_user_verified": ret_user_verified,
                    "retweet_user_created_at": ret_user_created_at,
                    "retweet_user_tweet_created_at": ret_tweet_date
                }
            }
        
        else: # normal tweet (without retweet and quote)
            if 'extended_tweet' in record_json_keys:
                text = record_json['extended_tweet']['full_text']
            else:
                text = record_json['text']

    return {
        tweet_id: {
            "tweet": text,
            "sentiment": '',
            "tweet_created_at": tweet_date,
            "user_id": user_id,
            "user_name": user_name,
            "user_location": user_location,
            "user_followers": user_followers,
            "user_friends": user_friends,
            "user_verified": user_verified,
            "user_created_at": user_created_at
        }
    }

def pre_process_tweet(text):
    # remove mentions (e.g., @realDonaldTrump) 
    # remove URLs (e.g., https://t.co/Oh2IsZ6JAw)
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    
    # remove 'RT' 
    text = text.replace("RT", "")

    # remove '#'
    text = text.replace("#", "")

    # convert emoji to text
    text = emoji.demojize(text, delimiters=("", ""))

    # replace '_' by ' ' (created by emojis)
    text = text.replace("_", " ")

    return text

def batch_tweets_dict(records):
    trump_tweets = {}
    biden_tweets = {}
    trump_biden_tweets = {}

    for record in records:
        if record.error():
            logging.error(
                'CONSUME (batch_tweets_dict): # %s - Consumer ERROR: %s', 
                os.getpid(), record.error()
            )
            continue
        
        if record.value() is None:
            continue

        try: 
            # convert bytes to str
            record_str = record.value().decode('utf-8') 
            tweet_info = extract_tweet_info(record_str)

        except InvalidTweet as e: 
            continue

        except NotEnglishTweet:
            continue

        except json.decoder.JSONDecodeError: 
            continue
        
        flag_trump = False
        flag_biden = False
        tweet_id = next(iter(tweet_info))
        tweet = tweet_info[tweet_id].get('tweet')

        if re.search('trump', tweet, re.IGNORECASE):
            flag_trump = True

        if re.search('biden', tweet, re.IGNORECASE):
            flag_biden = True

        if flag_trump or flag_biden: # apply pre process of tweet if has trump or biden
            tweet = pre_process_tweet(tweet)
            tweet_info[tweet_id]['tweet'] = tweet
        
        if flag_trump and flag_biden:
            trump_biden_tweets.update(tweet_info)

        elif flag_trump:
            trump_tweets.update(tweet_info)

        elif flag_biden:
            biden_tweets.update(tweet_info)

        else:
            continue
    logging.info(
        'CONSUME (batch_tweets_dict): #%s - Found - Trump tweets: %d; Biden tweets: %d; Trump & Biden tweets: %d',
         os.getpid(), len(trump_tweets), len(biden_tweets), len(trump_biden_tweets)
    )

    return {
        "trump": trump_tweets,
        "biden": biden_tweets,
        "trump_biden":  trump_biden_tweets
    }

def bulk_tweets(index, candidate_tweets):
    for tweet_id, tweet_info in candidate_tweets.items():
        if 'retweet_user_name' in tweet_info.keys():
            yield {
                "_index": '2020elections-'+index,
                "_id": tweet_id,
                "@timestamp": tweet_info.get("tweet_created_at"),
                "sentiment": class_names[tweet_info.get("sentiment")],
                "tweet": tweet_info.get("tweet"),
                "user_id": tweet_info.get("user_id"),
                "user_name": tweet_info.get("user_name"),
                "user_location": tweet_info.get("user_location"),
                "user_followers": tweet_info.get("user_followers"),
                "user_friends": tweet_info.get("user_friends"),
                "user_verified": tweet_info.get("user_verified"),
                "user_created_at": tweet_info.get("user_created_at"),
                "retweet_user_name": tweet_info.get("retweet_user_name"), 
                "retweet_user_location": tweet_info.get("retweet_user_location"), 
                "retweet_user_followers": tweet_info.get("retweet_user_followers"),
                "retweet_user_verified": tweet_info.get("retweet_user_verified"),
                "retweet_user_created_at": tweet_info.get("retweet_user_created_at"),
                "retweet_user_tweet_created_at": tweet_info.get("retweet_user_tweet_created_at")
            }

        else:
            yield {
                "_index": '2020elections-'+index,
                "_id": tweet_id,
                "@timestamp": tweet_info.get("tweet_created_at"),
                "sentiment": class_names[tweet_info.get("sentiment")],
                "tweet": tweet_info.get("tweet"),
                "user_id": tweet_info.get("user_id"),
                "user_name": tweet_info.get("user_name"),
                "user_location": tweet_info.get("user_location"),
                "user_followers": tweet_info.get("user_followers"),
                "user_friends": tweet_info.get("user_friends"),
                "user_verified": tweet_info.get("user_verified"),
                "user_created_at": tweet_info.get("user_created_at")
            }

def _init_sentiment_classifier(model_name, model_path):
    if model_name == 'bert':
        # load BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # load BERT sentiment classifier
        logging.info(
            'CONSUME (init model):#%s - Loading fine-tuned %s...', 
            os.getpid(), model_name
        )
        
        fine_tuned_weights = torch.load(model_path, map_location=torch.device("cpu"))
        model = BERT(CLASSES)
        model.load_state_dict(fine_tuned_weights, strict=False)
        sentimentclassifier = BERTSentimentClassifier(model, tokenizer, NUM_THREADS)
        
        logging.info(
            'CONSUME (init model):#%s - %s has been loaded', 
            os.getpid(), model_name
        )
    
    elif model_name == 'distillbert':
        # load distillbert tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # load DistillBERT sentiment classifier
        logging.info(
            'CONSUME (init model):#%s - Loading fine-tuned %s...', 
            os.getpid(), model_name
        )
        fine_tuned_weights = torch.load(model_path, map_location=torch.device("cpu"))
        model = DistillBERT(CLASSES)
        model.load_state_dict(fine_tuned_weights, strict=False)
        sentimentclassifier = DistillBERTSentimentClassifier(model, tokenizer, NUM_THREADS)
        
        logging.info(
            'CONSUME (init model):#%s - %s has been loaded', 
            os.getpid(), model_name
        )
    else:
        logging.critical(
            "CONSUME (init model):#%s Input model not correct. Please insert 'bert' or 'distillbert'",
            os.getpid()
        )
        exit(1)

    # if there's a GPU available...
    if torch.cuda.is_available() and USE_GPU:    
        sentimentclassifier.move_model_gpu()
        logging.info(
            'CONSUME (init model):#%s - There are %d GPU(s) available.', 
            os.getpid(), torch.cuda.device_count()
        )
        logging.info(
            'CONSUME (init model):#%s - %s moved to the GPU: %s', 
            os.getpid(), model_name, torch.cuda.get_device_name(0)
        )

    else:
        sentimentclassifier.move_model_cpu()
        logging.info(
            'CONSUME (init model):#%s - %s moved to the CPU.', 
            os.getpid(), model_name
        )
    return sentimentclassifier


def _consume(config, model, model_path):
    logging.info(
        'CONSUME: #%s - Starting consumer group=%s, topic=%s',
        os.getpid(), config['kafka_kwargs']['group.id'], config['topic'],
    )
    c = Consumer(**config['kafka_kwargs'])
    c.subscribe([config['topic']])

    sentimentclassifier = _init_sentiment_classifier(model, model_path)
    barrier.wait()

    while True:
        logging.info(
            'CONSUME: #%s - Waiting for message...', 
            os.getpid()
        )
        try:
            records = c.consume(num_messages=MAX_POLL_RECORDS, 
                                timeout=MAX_BLOCK_WAIT_TIME)

            logging.info(
                'CONSUME: #%s - Received %d records.',
                os.getpid(), len(records)
            )
            
            if len(records) == 0:
                time.sleep(2)
                continue
            
            # get batch of tweets (dict of tweets for trump, biden and both)
            batch_tweets = batch_tweets_dict(records)
            
            for index, candidate_tweets in batch_tweets.items():
                ids = []
                tweets = []
                for key in candidate_tweets:
                    ids.append(key)
                    tweets.append(candidate_tweets[key].get('tweet'))

                tweets_dataset = TweetsDataset(ids, tweets)
                tweets_dataloader = DataLoader(tweets_dataset, batch_size=BATCH_SIZE)
                
                # TODO: test GC
                # gc.collect()

                ids, predictions = sentimentclassifier.predict(tweets_dataloader)
                for id, sentiment in zip(ids, predictions):
                    candidate_tweets[id]['sentiment'] = sentiment


                # feed tweets to ElasticSearch
                try:
                    # no memory allocation when sending bulk of tweets to ES
                    response = helpers.bulk(es, bulk_tweets(index, candidate_tweets))

                    logging.info(
                        'CONSUME (to ElasticSearch): #%s - %s',
                        os.getpid(), str(response)
                    )
                except Exception as e:
                    logging.critical(
                        'CONSUME (to ElasticSearch EXCEPTION): #%s - %s', 
                        os.getpid(), str(e)
                    )

            try:
                c.commit()
                logging.info(
                    'CONSUME: #%s - Offsets have ben committed.',
                    os.getpid()
                )

            except Exception as e:
                logging.critical(
                    'CONSUME: #%s - Exception when committing offsets: %s', 
                    os.getpid(), str(e)
                ) 
            

        except KeyboardInterrupt: 
            logging.warning(
                'CONSUME: #%s - Worker is closing. Closing Kafka consumer gracefully...',
                os.getpid()
            )
            c.close()
            logging.warning(
                'CONSUME: #%s - Kafka consumer closed gracefully.',
                os.getpid()
            )
            sys.exit(0)

        except KafkaError as e: 
            logging.critical(
                'CONSUME: #%s - KAFKA CRITICAL ERROR: %s', 
                os.getpid(), str(e)
            )


def main():
    logging.basicConfig(format=LOGGING_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    
    consumer_config_path = sys.argv[1]
    model = sys.argv[2]
    model_path = sys.argv[3]

    with open(consumer_config_path) as f:
        configurations = json.load(f)

    kafka_conf = create_kafka_config(configurations)

    config={
        # At most, this should be the total number of Kafka partitions on
        # the topic.
        'num_workers': NUM_WORKERS,
        'num_threads': NUM_THREADS,
        'topic': configurations.get('topic'),
        'kafka_kwargs': kafka_conf
    }

    workers = []
    while True:
        num_alive = len([w for w in workers if w.is_alive()])
        
        # TODO: check CPU usage
        if config['num_workers'] == num_alive:
            continue
        
        # TODO: in case of failure it connects two other processes
        for _ in range(config['num_workers']-num_alive):
            p = Process(target=_consume, daemon=True, args=(config, model, model_path))
            p.start()
            workers.append(p)
            logging.info(
                'MAIN: Starting worker #%s', 
                p.pid
            )


if __name__ == "__main__":
    es = Elasticsearch(hosts="localhost", maxsize=NUM_WORKERS)
    main()