import com.google.common.collect.Lists;
import com.twitter.hbc.ClientBuilder;
import com.twitter.hbc.core.Client;
import com.twitter.hbc.core.Constants;
import com.twitter.hbc.core.Hosts;
import com.twitter.hbc.core.HttpHosts;
import com.twitter.hbc.core.endpoint.StatusesFilterEndpoint;
import com.twitter.hbc.core.processor.StringDelimitedProcessor;
import com.twitter.hbc.httpclient.auth.Authentication;
import com.twitter.hbc.httpclient.auth.OAuth1;
import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class TwitterProducer {
    private String consumerKey;
    private String consumerSecret;
    private String token;
    private String secret;
    final private Logger logger = LoggerFactory.getLogger(TwitterProducer.class.getName());

    ArrayList<String> termToSearch = new ArrayList<String>();

    private String bootstrapservers;
    private String topic;

    public TwitterProducer(JSONObject jsonObj){
        consumerKey = (String) jsonObj.get("consumerKey");
        consumerSecret = (String) jsonObj.get("consumerSecret");
        token = (String) jsonObj.get("token");
        secret = (String) jsonObj.get("secret");

        JSONArray jArray = (JSONArray)jsonObj.get("terms"); // ['trump', 'donald trump', 'biden', 'joe biden']
        if (jArray != null) {
            for (Object o : jArray) {
                termToSearch.add(o.toString());
            }
        }

        bootstrapservers = (String) jsonObj.get("bootstrapservers"); // 127.0.0.1:9092
        topic = (String) jsonObj.get("topic"); // elections_test
    }

    public static void main(String[] args) {
        String path = args[0];
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(path));
            new TwitterProducer((JSONObject) obj).run();

        } catch (ParseException | IOException e) {
            e.printStackTrace();
        }
    }

    private void run() {
        logger.info("SETUP");

        // Set up your blocking queues: Be sure to size these properly based on expected TPS of your stream
        BlockingQueue<String> msgQueue = new LinkedBlockingQueue<String>(1000); // capacity of # messages

        // create a twitter client
        Client client = createTwitterClient(msgQueue);

        // Attempts to establish a connection.
        client.connect();

        // create a kafka producer
        KafkaProducer<String, String> producer = createKafkaProducer();

        // add a shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread ( () -> {
            logger.info("Stopping application...");
            logger.info("Shutting down client from twitter...");
            client.stop();
            logger.info("Closing producer...");
            producer.close();
            logger.info("Done!");
        }));

        // loop to send tweets to kafka
        // on a different thread, or multiple different threads....
        while (!client.isDone()) {
            String msg = null;
            try {
                msg = msgQueue.poll(3, TimeUnit.SECONDS);

                ProducerRecord<String, String> record =
                        new ProducerRecord<String, String>(topic, null, msg);

                producer.send(record, new Callback() {
                    public void onCompletion(RecordMetadata recordMetadata, Exception e) {
                        // executes every time a record is sucessfully sent or an Exception is thrown
                        if (e == null) {
                            // record was sucessfully sent
                            //writeMetadataToLog(recordMetadata.topic(), recordMetadata.partition(),
                             //       recordMetadata.offset(), recordMetadata.timestamp());
                        } else {
                            logger.error("Error while producing", e);
                        }
                    }
                });
            } catch (InterruptedException e) {
                e.printStackTrace();
                client.stop();
            }
            if (msg != null){
                logger.info("Received a new tweet");
            }
        }

        logger.info("End of application.");
    }

    private Client createTwitterClient(BlockingQueue<String> msgQueue){
        // Declare the host you want to connect to, the endpoint, and authentication (basic auth or oauth)
        Hosts hosebirdHosts = new HttpHosts(Constants.STREAM_HOST);
        StatusesFilterEndpoint hosebirdEndpoint = new StatusesFilterEndpoint();
        // Optional: set up some followings and track terms
        //  List<Long> followings = Lists.newArrayList(1234L, 566788L); people
        List<String> terms = Lists.newArrayList(termToSearch); // terms
        // hosebirdEndpoint.followings(followings); people
        hosebirdEndpoint.trackTerms(terms); // terms

        // These secrets should be read from a config file
        Authentication hosebirdAuth = new OAuth1(consumerKey, consumerSecret, token, secret);

        ClientBuilder builder = new ClientBuilder()
                .name("Hosebird-Client-01")                              // optional: mainly for the logs
                .hosts(hosebirdHosts) // connect to STREAM HOST
                .authentication(hosebirdAuth) // use hosebirdAuth (with all keys)
                .endpoint(hosebirdEndpoint) // use a StatusesFilterEndpoint which tracks terms in this case
                .processor(new StringDelimitedProcessor(msgQueue));

        Client hosebirdClient = builder.build();
        return hosebirdClient;
    }

    private KafkaProducer<String, String> createKafkaProducer(){
        Properties properties = new Properties();
        properties.setProperty(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapservers);
        properties.setProperty(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        properties.setProperty(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        // create safe producer
        properties.setProperty(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, "true");
        // not needed but to be explicit !
        properties.setProperty(ProducerConfig.ACKS_CONFIG, "all");
        properties.setProperty(ProducerConfig.RETRIES_CONFIG, Integer.toString(Integer.MAX_VALUE));
        properties.setProperty(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, "5"); // version 2.5.0 kafka

        // high throughput producer
        properties.setProperty(ProducerConfig.COMPRESSION_TYPE_CONFIG, "snappy");
        properties.setProperty(ProducerConfig.LINGER_MS_CONFIG, "20");
        properties.setProperty(ProducerConfig.BATCH_SIZE_CONFIG, Integer.toString(64*1024)); // 64 KB batch size


        KafkaProducer<String, String> producer = new KafkaProducer<String, String>(properties);
        return producer;
    }

    private void writeMetadataToLog(String topic, int partition, long offset, long timestamp){
        logger.info("Received new TWEET to KAFKA: \n" +
                "Topic: " + topic + "\n" +
                "Partition: " + partition + "\n" +
                "Offset: " + offset + "\n" +
                "Timestamp: " + timestamp);
    }
}

