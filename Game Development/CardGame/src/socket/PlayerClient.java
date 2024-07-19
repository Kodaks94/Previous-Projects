package socket;

import assign.Stock;
import gui.FrameControl;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Scanner;


/**
 * Created by mpishe on 22/11/2017.
 */
public class PlayerClient {
    public Scanner in;
    public PrintWriter out;
    public Socket socket;
    public int playerID;
    public PlayerClient() {
        try{
            socket = new Socket(SocketConstant.SERVER, SocketConstant.PORT);
            InputStream instream = socket.getInputStream();
            OutputStream outstream = socket.getOutputStream();
            in = new Scanner(instream);
            out = new PrintWriter(outstream, true);//auto flush
            playerID=AddPlayer();
        }catch (IOException e){
            e.printStackTrace();
        }
    }
    public String Command(String command){
        StringBuffer result = new StringBuffer();
        out.println(command);
        while(true){
            String line = in.nextLine().trim();
            if(line.isEmpty()) break;
                result.append(line);

        }
        return result.toString();
    }
    public int AddPlayer(){
        String response = Command("add");
        return Integer.parseInt(response);
    }
    //GETCASH
    public int GetCash(int i){
        String response = Command("getcash "+i);
        return Integer.parseInt(response);
    }

    //GETROUND
    public int GetRound(){
        String response = Command("getround");
        return Integer.parseInt(response);
    }
    //SETROUND
    public int SetRound(int i){
        String response = Command("setround "+i);
        return Integer.parseInt(response);
    }
    public String buy(int id, Stock s, int amount){
        String response = Command("buy "+id+" "+s+" "+amount);
        return response;
    }
    //SELL
    public String sell(int id , Stock s , int amount ){
        String response =  Command("sell "+id+" "+s+" "+amount);
        return response;
    }
    //VOTE
    public String vote(int id , Stock s, boolean yn, boolean yes){
        String response = Command("vote "+id+" "+s+" "+Boolean.toString(yn)+" "+Boolean.toString(yes));
        return response;
    }
    //GETPRICES
    public int[] getPrices(){
        String response = Command("getprices");
       return StringtoIntArr(response);
    }
    //String to int array
    private int[] StringtoIntArr(String response){
        response = response.substring(1,response.length() -1);
        String[] ArrString = response.split(",(\\s*)");
        int[] Values = new int[ArrString.length];
        for (int i = 0; i < Values.length; i++) {
            Values[i] = Integer.parseInt(ArrString[i]);
        }
        return Values;
    }
    //Finished
    public String Finished(int id){
        String response  = Command("finished "+id);
        return response;

    }
    //CHECK
    public boolean playersCheck(){
        String response = Command("check");
        return Boolean.parseBoolean(response);
    }
    //GETCARDS
   public int[] getCards(){
       String response= Command("getcards");
       return StringtoIntArr(response);
   }
   //GETSHARE
    public int[] getShare(int id){
        String response = Command("getshare "+id);
        return StringtoIntArr(response);

    }
    //GetResults
    public String Getresult(){
        String response = Command("getresult");
        return response;
    }
    //ADDMUL
    public String ADDMUL(){
       String response = Command("addmul");
       return response;
    }

}
