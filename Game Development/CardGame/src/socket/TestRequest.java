package socket;

import com.sun.org.apache.regexp.internal.RE;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestRequest {

    String[] requestStrings = new String[]{"GETCASH 1","VOTE 1 APPLE TRUE FALSE","SELL 1 APPLE 5","BUY 1 APPLE 5","GETSHARE 1","GETROUND","SETROUND 2","SETCASH 200","GETCARDS","ADD","GETPRICES"};

    public TestRequest(){

    };
    @Test
    public void stringToRequestString(){
        for (String requestString: requestStrings){
            assertEquals(requestString,Request.parse(requestString).toString());
        }
    }
    @Test
    public void requestToStringToRequest() {
        for (String requestString : requestStrings) {
            Request request = Request.parse(requestString);
            assertEquals(request, Request.parse(request.toString()));
        }
    }

}
