# Pipeline-US-2020-Presidential-Elections

Ongoing project. 
Perform sentiment analysis classification over realtime tweets of the principal candidates of 2020 US Presidential elections.

## Log

October 1, 2020: Python Kafka consumer has been implemented and is able to consume batches of messages. Moreover, some messages are extended tweets (text has than 140 characters) and others are considered bad data (e.g., it was received a JSON like {"limit":{"track":506,"timestamp_ms":"1601553530778"}})

September 26, 2020: Java Kafka Producer has been connected to a Java HTTP client for Twitter tweet streaming (https://github.com/twitter/hbc). The implemented producer is able to achieve high throughput performance by enabled compression and batching of messages.

September 16, 2020: Sentiment analysis classifier has been implemented with a BERT base model. Fine-tune of BERT performed in Google Colab with GPU access.
