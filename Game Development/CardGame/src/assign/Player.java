package assign;

import javax.xml.bind.SchemaOutputResolver;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by mpishe on 21/11/2017.
 */
public class Player {
    public final int playerID;
    private int cash;
    public  int[] shares = new int[5];
    public boolean finished;
    public Player(int playerID) {
        this.playerID = playerID;
        this.cash = Constants.INITIALCASH;
        InitialShareSet();
        finished = false;
    }

    public void InitialShareSet() {
        Random rn = new Random();
        int sum = 0;
        List<Integer> visited = new ArrayList<Integer>();
        while(sum !=10) {
            int nextShare = rn.nextInt(5);
            if(!visited.contains(nextShare)) {
                visited.add(nextShare);
                int add = rn.nextInt(Math.abs(Constants.INITIALSHARE + 1 - sum));
                sum += add;
                shares[nextShare] += add;
            }
            if(visited.size() == 4) {
                visited.clear();
            }


        }

    }
    public int GetCash(){
        return cash;
    }
    public int[] GetShares(){
        return shares;
    }
    public void setCash(int givenCash){
        this.cash = givenCash;
    }
    public void setShare(int indexStock, int amount){
        shares[indexStock] = amount;
    }
    @Override
    public String toString() {
        return String.format("Player [%d, %d, %s]", playerID, cash, Arrays.toString(shares));
    }



}


