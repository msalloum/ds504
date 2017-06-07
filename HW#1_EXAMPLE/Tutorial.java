import org.apache.spark.*;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.streaming.*;
import org.apache.spark.streaming.twitter.*;
import org.apache.spark.streaming.api.java.*;
import twitter4j.*;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.*;


public class Tutorial {

    static ArrayList<String> s = new ArrayList();
    static int numberInst = 0;
    static final int BUCKETSPERELEMENT = 10;
    static final int ELEMENTS = 10000;

    static BloomFilter filter = new BloomFilter(
        ELEMENTS, BUCKETSPERELEMENT);

  public static void main(String[] args) throws Exception {

    Logger.getLogger("org").setLevel(Level.ERROR);
    Logger.getLogger("akka").setLevel(Level.ERROR);
   
    // Configuring Twitter credentials 
    String consumerKey = "FgXImus1bXVMREkethe8Q";
    String consumerSecret = "Xh6T2pzZRDis3tpgtfHL5SZLksWwwRQB1JvUiImUPI";
    String accessToken = "2193013314-xo7pCQ7lnI4aGnTRlCTIKnWsLBRDby26WcpyW2U";
    String accessTokenSecret = "wmq3or8cS5XrrWwqCskbzefQ7UAzHzC7P9TvF4bAK0TQk";

    SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("SparkTwitter");
    JavaStreamingContext jssc = new JavaStreamingContext(conf, new Duration(10000));

    System.setProperty("twitter4j.oauth.consumerKey", consumerKey);
    System.setProperty("twitter4j.oauth.consumerSecret", consumerSecret);
    System.setProperty("twitter4j.oauth.accessToken", accessToken);
    System.setProperty("twitter4j.oauth.accessTokenSecret", accessTokenSecret);

    JavaReceiverInputDStream<Status> twitterStream = TwitterUtils.createStream(jssc);

    Pattern MY_PATTERN = Pattern.compile("#(\\S+)");
 

    // With filter: Only use tweets with geolocation and print location+text.
    /*JavaDStream<Status> tweetsWithLocation = twitterStream.filter(
                
                new Function<Status, Boolean>() {
                    public Boolean call(Status status){
                        if (status.getGeoLocation() != null) {
                            return true;
                        } else {
                            return false;
                        }
                    }
                }
    );*/

    /* contains hashtags */ 
    JavaDStream<Status> TweetsWHashtags = twitterStream.filter(
 
        new Function<Status, Boolean>() {
            public Boolean call(Status status){
                if (status.getText().contains("#") ) {
                    return true;
                } else {
                    return false;
                }
            }
        }
    );

    JavaDStream<String> statuses = TweetsWHashtags.map(

        new Function<Status, String>() {
            public String call(Status status) {
                numberInst+=1;
                String text = status.getText();
                Matcher m = MY_PATTERN.matcher(text);
                String tag = "";
                
                if(m.find()) {
                    tag = m.group(1);
                    filter.add(tag);
                    s.add(tag);
                } 
                return "#"+tag;
            }
        }
    );

    statuses.print();
    jssc.start();

    while (numberInst < 1000){
        jssc.awaitTerminationOrTimeout(1000); // 1 second polling time, you can change it as per your usecase
    }

    jssc.stop();
    System.out.println("Test if 'WeAreUK' in stream: "  + filter.contains("WeAreUK"));
    System.out.println("Test if '#DS504' in stream: " + filter.contains("DS504"));

}

}
