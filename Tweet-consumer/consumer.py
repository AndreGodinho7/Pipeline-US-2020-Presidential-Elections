import json
import logging
from confluent_kafka import Consumer, KafkaError

# Logging configs
LOGGING_FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'

# Kafka configs
CONSUMER_CONFIG_PATH = '/home/andregodinho06/Projects/Twitter Project/consumer.json'
AUTO_OFFSET_RESET = 'earliest'
ENABLE_AUTO_COMMIT = False

# maximum number of messages to return
MAX_POLL_RECORDS = 100 # TODO: how many records to poll?

# maximum time to block waiting for message (in seconds)
MAX_BLOCK_WAIT_TIME = 2

# maximum amount of time between two .poll() calls before declaring the consumer dead
POLL_INTERVAL_MIN = 20
MAX_POLL_INTERVAL_MS = POLL_INTERVAL_MIN*60*1000 # TODO: how many milliseconds to wait between poll?


def create_kafka_consumer(jsonData):
    conf = {
        'bootstrap.servers': jsonData.get('bootstrapservers'),
        'group.id': jsonData.get('groupid'),
        'auto.offset.reset': AUTO_OFFSET_RESET,
        'enable.auto.commit': ENABLE_AUTO_COMMIT,
        'max.poll.interval.ms' : MAX_POLL_INTERVAL_MS
    }

    kafkaConsumer = Consumer(conf)
    kafkaConsumer.subscribe([jsonData.get('topic')])

    return kafkaConsumer

def extract_twitter_id_text(record):
    record_json = json.loads(record)

    if 'extended_tweet' in record_json.keys():
        return record_json['id_str'], record_json['extended_tweet']['full_text']
    
    return record_json['id_str'], record_json['text']






def main():
    logging.basicConfig(format=LOGGING_FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    with open(CONSUMER_CONFIG_PATH) as f:
        configurations = json.load(f)

    kafkaConsumer = create_kafka_consumer(configurations)
    logging.info("Created Kafka Consumer.")

    # poll for new data
    while(True):
        try: 
            records = kafkaConsumer.consume(num_messages=MAX_POLL_RECORDS, 
                                            timeout=MAX_BLOCK_WAIT_TIME)
            
            logging.info(f'Received {len(records)} records.')
            
            for record in records:
                record_str = record.value().decode('utf-8')
                
                try:
                    tweet_id, tweet_text = extract_twitter_id_text(record_str)
                    # print(f'Tweet ID: {tweet_id}')
                    # print(f'Tweet text: {tweet_text}\n\n')

                except:
                    logging.warning(f'Skipping bad data {record_str}')

            try: 
                if (len(records) > 0):
                    logging.info("Commiting consumer offsets...")
                    kafkaConsumer.commit()
                    logging.info("Offsets have been committed.")

            except Exception as e:
                logging.critical(f'Thown exception when committing offsets {e}')

        except KeyboardInterrupt: 
            logging.warning('Python program is closing. Closing Kafka consumer gracefully...')
            kafkaConsumer.close()
            logging.warning('Kafka consumer closed gracefully.')
            exit(0)

        except KafkaError as e: 
            logging.critical(f"KAFKA INTERNAL ERROR {e}")
    

if __name__ == "__main__":
    main()