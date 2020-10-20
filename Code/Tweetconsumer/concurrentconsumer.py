######## UTILS CONFIGS ########
import json
import time
import sys
import os
sys.path.append("..") # to be able to import modules




######## LOGGING CONFIGS ########
import logging
LOGGING_FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'




######## KAFKA CONFIGS ########
from confluent_kafka import Consumer, KafkaError


## CONSUMER_CONFIG_PATH = '/home/andregodinho06/Projects/Twitter Project/consumer.json'
AUTO_OFFSET_RESET = 'earliest'
ENABLE_AUTO_COMMIT = False

## maximum number of messages to return
MAX_POLL_RECORDS = 100 # TODO: how many records to poll?

## maximum time to block waiting for message (in seconds)
MAX_BLOCK_WAIT_TIME = 2

## maximum amount of time between two .poll() calls before declaring the consumer dead
POLL_INTERVAL_MIN = 20
MAX_POLL_INTERVAL_MS = POLL_INTERVAL_MIN*60*1000 # TODO: how many milliseconds to wait between poll?




######## MODEL CONFIGS ########
import numpy as np
import torch
from transformers import BertTokenizer, DistilBertTokenizer
from BERTSentimentAnalysis.DataPreProcess.BERTFormatDataloader import BERTFormatDataloader
from BERTSentimentAnalysis.DataPreProcess.BERTInferenceDataset import BERTInferenceDataset
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
CLASSES = 3
class_names = ['negative', 'neutral', 'positive']
USE_GPU = False
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
#MODEL_NAME = '/home/andregodinho06/Projects/Twitter Project/bert_classifier.bin'
#MODEL_NAME = '/home/andregodinho06/Projects/Twitter Project/distillbert_classifier.bin'


######## WORKER CONFIGS ########
import threading
threading.stack_size(10000*1024*1024)
from multiprocessing import Process
from queue import Queue


NUM_WORKERS = 1
NUM_THREADS = 1

def create_kafka_config(jsonData):
    conf = {
        'bootstrap.servers': jsonData.get('bootstrapservers'),
        'group.id': jsonData.get('groupid'),
        'auto.offset.reset': AUTO_OFFSET_RESET, # 'earliest'
        'enable.auto.commit': ENABLE_AUTO_COMMIT, # False
        'max.poll.interval.ms' : MAX_POLL_INTERVAL_MS #TODO: tune (now it's 20 min)
    }

    return conf

def extract_twitter_id_text(record):
    record_json = json.loads(record)

    if 'extended_tweet' in record_json.keys():
        return record_json['id_str'], record_json['extended_tweet']['full_text']
    
    return record_json['id_str'], record_json['text']

def batch_tweets_dict(records):
    batch_dict = {}
    for record in records:
        if record.error():
            logging.error(
                'CONSUME (batch_tweets_dict):# %s - Consumer ERROR: %s', 
                os.getpid(), record.error()
            )
            continue

        # convert bytes to str
        record_str = record.value().decode('utf-8') 
        try: 
            id_tweet, text_tweet = extract_twitter_id_text(record_str)
        except KeyError: 
            logging.warning('Skipping bad data: ' +record_str)
        except json.decoder.JSONDecodeError: 
            logging.warning('Skipping bad data: ' +record_str)
            continue
        
        batch_dict.update({id_tweet:text_tweet})
        
    return batch_dict

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
        logging.ERROR("Input model not correct. Please insert 'bert' or 'distillbert'")
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



def _process_batch(q, c, model, model_path):
    batch = q.get(timeout=60)  # Set timeout to care for POSIX<3.0 and Windows.
    
    logging.info(
        'CONSUME (process batch): #%s THREAD#%s - Received %d records.',
        os.getpid(), threading.get_ident(), len(batch)
    )

    # batch needs to be in np.ndarray format for batches of dataloader
    batch = np.array(list(batch.values()))
    
    sentimentclassifier = _init_sentiment_classifier(model, model_path)

    start = time.process_time()
    predictions = sentimentclassifier.predict(batch, BATCH_SIZE)
    print(predictions, flush=True)
    total_time = round(time.process_time() - start, 2)

    logging.info(
        'CONSUME (process batch): #%s THREAD#%s - classification time = %f',
        os.getpid(), threading.get_ident(), total_time
    )

    q.task_done()
    exit(0)
    try:
        c.commit()

    except Exception as e:
        logging.critical(
            'CONSUME (process batch): #%s THREAD#%s - Exception when committing offsets: %s', 
            os.getpid(), threading.get_ident(), str(e)
        ) 


def _consume(config, model, model_path):
    logging.info(
        'CONSUME: #%s - Starting consumer group=%s, topic=%s',
        os.getpid(), config['kafka_kwargs']['group.id'], config['topic'],
    )
    c = Consumer(**config['kafka_kwargs'])
    c.subscribe([config['topic']])
    q = Queue(maxsize=config['num_threads'])

    # sentimentclassifier = _init_sentiment_classifier(model, model_path)

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
                continue
            
            # get batch of tweets in a dict {tweet ID: tweet text} (able to get long tweets)
            batch_records = batch_tweets_dict(records)

            q.put(batch_records)

            # Use default daemon=False to stop threads gracefully in order to
            # release resources properly.
            t = threading.Thread(target=_process_batch, args=(q, c, model, model_path))
            t.start()
        
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
            exit(0)

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

        for _ in range(config['num_workers']-num_alive):
            p = Process(target=_consume, daemon=True, args=(config, model, model_path))
            p.start()
            workers.append(p)
            logging.info(
                'MAIN: Starting worker #%s', 
                p.pid
            )


if __name__ == "__main__":
    main()