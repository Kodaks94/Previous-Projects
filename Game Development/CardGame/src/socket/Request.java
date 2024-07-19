package socket;

import java.util.Arrays;

/**
 * Created by mpishe on 22/11/2017.
 */
public class Request {

    public RequestType rt;
    public String[] params;

    public Request(RequestType rt, String...params){
        this.rt = rt;
        this.params = params;
    }
    @Override
    public String toString(){
        StringBuilder b = new StringBuilder();
        b.append(rt);
        for(int i = 0; i < params.length; i++){
            b.append(" "+params[i]);
        }
        return b.toString();
    }
    public static Request parse(String command){
        try{
            String[] commands = command.trim().split("\\s+");
            switch (commands[0].toUpperCase()){
                case "VOTE":
                    return new Request(RequestType.VOTE,commands[1],commands[2],commands[3],commands[4]);
                case "BUY":
                    return new Request(RequestType.BUY,commands[1],commands[2],commands[3]);
                case "SELL":
                    return new Request(RequestType.SELL,commands[1],commands[2],commands[3]);
                case "GETCASH":
                    return new Request(RequestType.GETCASH,commands[1]);
                case "GETSHARE":
                    return new Request(RequestType.GETSHARE,commands[1]);
                case "GETROUND":
                    return new Request(RequestType.GETROUND);
                case "SETROUND":
                    return new Request(RequestType.SETROUND,commands[1]);
                case "SETCASH":
                    return new Request(RequestType.SETCASH, commands[1]);
                case "GETCARDS":
                    return new Request(RequestType.GETCARDS);
                case "GETPRICES":
                    return new Request(RequestType.GETPRICES);
                case "ADD":
                    return new Request(RequestType.ADD);
                case "CHECK":
                    return new Request(RequestType.CHECK);
                case "FINISHED":
                    return new Request(RequestType.FINISHED, commands[1]);
                case "GETRESULT":
                    return new Request(RequestType.GETRESULT);
                case "ADDMUL":
                    return new Request(RequestType.ADDMUL);
                case "GETGAME":
                    return new Request(RequestType.GETGAME);
                default:
            }
        } catch (ArrayIndexOutOfBoundsException e){
        }
         return new Request(RequestType.INVALID);
    }
    @Override
    public boolean equals(Object obj){
        if (this == obj) return true;
        if (obj == null) return false;
        if (this.getClass() != obj.getClass()) return false;
        Request other = (Request) obj;
        if (!Arrays.equals(params, other.params)) return  false;
        if(rt != other.rt) return false;
        return true;
    }
}
