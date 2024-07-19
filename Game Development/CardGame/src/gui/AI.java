package gui;

import assign.Game;
import assign.Player;
import assign.Stock;
import socket.PlayerClient;

import java.util.Arrays;

public class AI {
    public int Cash;
    public int Round;
    public int PhaseCounter;
    public int[] shares;
    public int[] prices;
    public int PlayerID;
    Boolean finished;
    public PlayerClient pc;
    public AI(){

        pc = new PlayerClient();
        PlayerID = pc.playerID;
        Round = pc.GetRound();
        Cash = pc.GetCash(PlayerID);
        shares = pc.getShare(PlayerID);
        prices = pc.getPrices();
        PhaseCounter = 1;
    }
    public void NetworkUPDATE(){
        Cash = pc.GetCash(PlayerID);
        Round = pc.GetRound();
        shares = pc.getShare(PlayerID);
        prices = pc.getPrices();
    }
    private String stats(){
        return " PlayerID "+PlayerID+" Shares "+ Arrays.toString(shares)+ " Cash: "+Cash+ " Prices "+ Arrays.toString( prices);
    }
    public void Update(){

        if(Round <=4){
            switch(PhaseCounter)
            {
                case 1:
                    NetworkUPDATE();
                    BuyAI();
                    NetworkUPDATE();
                case 2:
                    SellAI();
                    NetworkUPDATE();
                case 3:
                    VoteAI();
                    NetworkUPDATE();
                case 4:
                    Checking();
                    NetworkUPDATE();



            }



        }


    }
    private int maxSHARE(){
        int maxshare = Integer.MIN_VALUE;
        for(int i = 0; i < shares.length ; i++){
            if(shares[i] > maxshare){
                maxshare = i;

            }
        }
        return maxshare;
    }
    public void BuyAI(){
        System.out.println(stats());
        int stock = 0;
        int Mini= Integer.MAX_VALUE;
            for (int i = 0; i < prices.length; i++) {
                if (prices[i] < Mini && Cash > prices[i] && maxSHARE() == i ) {
                    Mini = prices[0];
                    stock = i;
                }
            }
            int amount = 1;
            while ((Mini * amount + 3*amount) <= (Cash * (60 / 100))) {
                amount++;
            }

            if (Mini != Integer.MAX_VALUE) {
                System.out.println("Buy "+ pc.buy(PlayerID, Stock.values()[stock], amount));
            }
        NetworkUPDATE();
        System.out.println(pc.GetCash(PlayerID));
        PhaseCounter++;
    }
    public void SellAI(){
        System.out.println(stats());
        int stock = 0;
        int Max = Integer.MIN_VALUE;

            for (int i = 0; i < prices.length; i++) {
                if (prices[i] > Max && pc.getShare(PlayerID)[i] > 0) {
                    Max = prices[i];
                    stock = i;
                }
            }
            int amount = 1;
            while (amount <= shares[stock]) {
                amount++;
            }
            amount-= 1;
            if (Max != Integer.MIN_VALUE) {
                System.out.println("Sell "+ pc.sell(PlayerID, Stock.values()[stock], amount));
            }
        NetworkUPDATE();
        PhaseCounter++;
    }
    public void VoteAI(){
        int stocklow = 0;
        int stockhigh = 0;
        int Max = Integer.MIN_VALUE;
        int Mini= Integer.MAX_VALUE;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] > Max && maxSHARE() == i) {
                Max = prices[i];
                stocklow = i;
            }
        }
        for (int i = 0; i < prices.length; i++ ) {
            if (prices[i] < Mini) {
                Mini = prices[i];
                stockhigh = i;
            }
        }
        int[] cards = pc.getCards();
        int choice = Integer.MIN_VALUE;
        int index =0;
        for(int i =0 ; i < 5; i++){
            if(cards[i] > choice && i == stockhigh){
                choice = cards[i];
                index = i;
            }
        }

        System.out.println("Vote UP "+pc.vote(PlayerID,Stock.values()[stockhigh],true, true));
        System.out.println( "Vote DOWN "+pc.vote(PlayerID, Stock.values()[stocklow],false, true ));
       finished = Boolean.valueOf(pc.Finished(PlayerID));
        NetworkUPDATE();
        PhaseCounter++;

    }
    public void Checking(){
        if(Round != pc.GetRound()){
            PhaseCounter = 1;
            Round = pc.GetRound();
        }
        NetworkUPDATE();
    }
}
