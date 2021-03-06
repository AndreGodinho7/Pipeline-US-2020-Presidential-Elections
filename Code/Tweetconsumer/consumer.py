import sys
sys.path.append("..")
import torch

from BERTSentimentAnalysis.DataPreProcess.BERTFormatDataloader import BERTFormatDataloader
from BERTSentimentAnalysis.DataPreProcess.BERTInferenceDataset import BERTInferenceDataset
# from BERTSentimentAnalysis.SentimentClassifier.BERTSentimentClassifier import BERTSentimentClassifier
# from BERTSentimentAnalysis.SentimentClassifier.BERT import BERT

from BERTSentimentAnalysis.SentimentClassifier.newBERTSentimentClassifier import BERTSentimentClassifier
from BERTSentimentAnalysis.SentimentClassifier.BERT import BERT
from BERTSentimentAnalysis.SentimentClassifier.DistillBERTSentimentClassifier import DistillBERTSentimentClassifier
from BERTSentimentAnalysis.SentimentClassifier.DistillBERT import DistillBERT

import json
import numpy as np
import torch
import logging
from confluent_kafka import Consumer, KafkaError
from transformers import BertTokenizer, DistilBertTokenizer
import time


# logging configs
LOGGING_FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'

# kafka configs
#CONSUMER_CONFIG_PATH = '/home/andregodinho06/Projects/Twitter Project/consumer.json'
AUTO_OFFSET_RESET = 'earliest'
ENABLE_AUTO_COMMIT = False

## maximum number of messages to return
MAX_POLL_RECORDS = 100 # TODO: how many records to poll?

## maximum time to block waiting for message (in seconds)
MAX_BLOCK_WAIT_TIME = 2

## maximum amount of time between two .poll() calls before declaring the consumer dead
POLL_INTERVAL_MIN = 20
MAX_POLL_INTERVAL_MS = POLL_INTERVAL_MIN*60*1000 # TODO: how many milliseconds to wait between poll?

# Model configs
RANDOM_SEED = 10
MAX_LEN = 512
BATCH_SIZE = 16
n_GPU = 0
CLASSES = 3
class_names = ['negative', 'neutral', 'positive']

NUM_THREADS = 1

#MODEL_NAME = '/home/andregodinho06/Projects/Twitter Project/bert_classifier.bin'
#MODEL_NAME = '/home/andregodinho06/Projects/Twitter Project/distillbert_classifier.bin'
USE_GPU = False

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def create_kafka_consumer(jsonData):
    conf = {
        'bootstrap.servers': jsonData.get('bootstrapservers'),
        'group.id': jsonData.get('groupid'),
        'auto.offset.reset': AUTO_OFFSET_RESET,
        'enable.auto.commit': ENABLE_AUTO_COMMIT,
        'max.poll.interval.ms' : MAX_POLL_INTERVAL_MS,
    }

    kafkaConsumer = Consumer(conf)
    kafkaConsumer.subscribe([jsonData.get('topic')])

    return kafkaConsumer

def extract_twitter_id_text(record):
    record_json = json.loads(record)

    if 'extended_tweet' in record_json.keys():
        return record_json['id_str'], record_json['extended_tweet']['full_text']
    
    return record_json['id_str'], record_json['text']

def batch_tweets_dict(records):
    batch_dict = {}
    for record in records:
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

def main():
    logging.basicConfig(format=LOGGING_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    
    consumer_config_path = sys.argv[1]
    model = sys.argv[2]
    model_path = sys.argv[3]

    with open(consumer_config_path) as f:
        configurations = json.load(f)

    kafkaConsumer = create_kafka_consumer(configurations)
    logging.info("Created Kafka Consumer.")

    if model == 'bert':
        # load BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # load BERT sentiment classifier
        logging.info("Loading fine-tuned model...")
        fine_tuned_weights = torch.load(model_path, map_location=torch.device("cpu"))
        model = BERT(CLASSES)
        model.load_state_dict(fine_tuned_weights, strict=False)
        sentimentclassifier = BERTSentimentClassifier(model, tokenizer, NUM_THREADS)
        logging.info("Model has been loaded.")
    
    elif model == 'distillbert':
        # load distillbert tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # load DistillBERT sentiment classifier
        logging.info("Loading fine-tuned model...")
        fine_tuned_weights = torch.load(model_path, map_location=torch.device("cpu"))
        model = DistillBERT(CLASSES)
        model.load_state_dict(fine_tuned_weights, strict=False)
        sentimentclassifier = DistillBERTSentimentClassifier(model, tokenizer, NUM_THREADS)
        logging.info("Model has been loaded.")

    else:
        logging.ERROR("Input model not correct. Please insert 'bert' or 'distillbert'")
        exit(1)


    # if there's a GPU available...
    if torch.cuda.is_available() and USE_GPU:    
        sentimentclassifier.move_model_gpu()
        logging.info('There are '+str(torch.cuda.device_count())+' GPU(s) available.')
        logging.info('Model moved to the GPU: '+str(torch.cuda.get_device_name(0)))

    else:
        # sentimentclassifier.move_model_cpu()
        logging.info('Model moved to the CPU.')

    # poll for new data
    while(True):
        try: 
            records = kafkaConsumer.consume(num_messages=MAX_POLL_RECORDS, 
                                            timeout=MAX_BLOCK_WAIT_TIME)
            
            logging.info('Received '+str(len(records))+' records.')
            
            # get batch of tweets in a dict {tweet ID: tweet text}
            batch_dict = batch_tweets_dict(records)
            
            # sentiment classification
            # batch_dataset = BERTInferenceDataset(list(batch_dict.values()), MAX_LEN, tokenizer)
            # batch_dataloader = BERTFormatDataloader(batch_dataset, BATCH_SIZE, n_GPU).getDataloader()

            print(list(batch_dict.keys()))

            logging.info("Classifying tweets...")
            start = time.process_time()
            predictions = sentimentclassifier.predict(np.array(list(batch_dict.values())), BATCH_SIZE)
            t = round(time.process_time() - start,2)
            print(t)

            # for record in records:
            #     record_str = record.value().decode('utf-8')

                # try:
                #     tweet_id, tweet_text = extract_twitter_id_text(record_str)
                #     print(f'Tweet ID: {tweet_id}')
                #     print(f'Tweet text: {tweet_text}\n\n')

                # except:
                #     logging.warning(f'Skipping bad data {record_str}')

            try: 
                if (len(records) > 0):
                    logging.info("Commiting consumer offsets...")
                    kafkaConsumer.commit()
                    logging.info("Offsets have been committed.")

            except Exception as e:
                logging.critical('Thown exception when committing offsets '+str(e))

            exit(0)


        except KeyboardInterrupt: 
            logging.warning('Python program is closing. Closing Kafka consumer gracefully...')
            kafkaConsumer.close()
            logging.warning('Kafka consumer closed gracefully.')
            exit(0)

        except KafkaError as e: 
            logging.critical('KAFKA INTERNAL ERROR '+str(e))
    

if __name__ == "__main__":
    main()
