package socket;

import assign.Game;
import assign.Player;
import assign.Stock;
import gui.FrameControl;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class GameService implements Runnable{

    private Scanner in;
    private PrintWriter out;
    private String command;
    private Game game;
    public GameService(Game game, Socket socket){
        this.game = game;
        this.command = null;
        try{
            in = new Scanner(socket.getInputStream());
            out = new PrintWriter(socket.getOutputStream(),true);
        }catch (IOException e){
            e.printStackTrace();
        }
    }
    @Override
    public void run(){
        while(true) {
            try {
                game.Update();
                Request request = Request.parse(in.nextLine());
                String responses = execute(game, request);
                // take actions execute...
                out.println(responses + "\r\n");

            } catch (NoSuchElementException e) {
                e.printStackTrace();
            }
        }
    }
    public boolean login(){
        if(Game.players.size() > 0){
            return true;
        }
        return false;
    }
    public String execute(Game game , Request request) {
        try {
            switch (request.rt) {
                case SELL:
                    return game.sell(Integer.parseInt(request.params[0]),Stock.valueOf(request.params[1]),Integer.parseInt(request.params[2]));
                case BUY:
                    return game.buy(Integer.parseInt(request.params[0]),Stock.valueOf(request.params[1]),Integer.parseInt(request.params[2]));
                case VOTE:
                    return game.vote(Integer.parseInt(request.params[0]), Stock.valueOf(request.params[1]),Boolean.valueOf(request.params[2]),Boolean.valueOf(request.params[3]));
                case INVALID:
                    break;
                case GETCASH:
                    return game.getCash(Integer.parseInt(request.params[0]));
                case GETROUND:
                    return String.valueOf(game.Round);
                case SETROUND:
                    game.Round = Integer.parseInt(request.params[0]);
                    return String.valueOf(game.Round);
                case GETPRICES:
                    return Arrays.toString(game.getPrices());
                case GETCARDS:
                    return Arrays.toString(game.getCards());
                case GETSHARE:
                    return Arrays.toString(game.getShares(Integer.parseInt(request.params[0])));
                case ADD:
                    String a = String.valueOf(game.addPlayer());
                    return a;
                case CHECK:
                    boolean returner = true;
                    for(Player p: game.players){
                        if(p.finished== false){
                            returner = false;
                        }
                    }
                    return String.valueOf(returner);
                case FINISHED:
                    game.IsFinished(Integer.parseInt(request.params[0]));
                    return "Changed";
                case GETRESULT:
                    String result = "";
                    int x = Integer.MIN_VALUE;
                    int pi= 0;
                    for(Player p : game.players){
                        result += "///Player: "+p.playerID+" Cash: "+p.GetCash()+ "\n";
                        if(p.GetCash() > x){ x = p.GetCash(); pi = p.playerID;}

                    }
                    result += "///Winner: Player "+pi+" Cash:"+x;
                    return result;
                case ADDMUL:
                   FrameControl frame1 = new FrameControl("The Stock Game", game);
                    FrameControl frame2 = new FrameControl("The Stock Game",game );
                    FrameControl frame3 = new FrameControl("The Stock Game",game );
                    frame2.setVisible(true);
                    frame1.setVisible(true);
                    frame3.setVisible(true);

                    return "";
                default:
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    return "";
    }

}
