package assign;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

public class Game {

	// skeleton class with dummy constructors and methods 
	public static List<Player> players;
	public static int[] StockPrices = {100,100,100,100,100};
	public static int[] Votes = {0,0,0,0,0};
	public static Deck[]deck = new Deck[5];
	public static int numofplayers = 0;
	public Card[] cards = new Card[5];
	public static int Round = 1;
	// create a random game 
	public Game() {
		players = new ArrayList<>();
		int n = 0;
		for (Stock s : Stock.values()){
			deck[n] = new Deck(s);
			n++;
		}
	}
	public int addPlayer(){
	    if(numofplayers < Constants.NUMBEROFPLAYERS){
            numofplayers++;
            Player p = new Player(numofplayers);
            players.add(p);
            return p.playerID;
        }
        else return 0;


	}
	// create a game with specific initial decks and share holdings
	// used for unit testing 
	public Game(Deck [] decks, int[][] shares) {
		this.deck = decks;

	}

	public String getCash(int playerId) {

		return String.valueOf(findPlayer(playerId).GetCash());

	}
	public Player findPlayer(int playerId) {
		for (Player p : players) {
			if (p.playerID == playerId) {
				return p;
			}
		}
		throw new NoSuchElementException();
	}
	public int[] getShares(int playerId) {
			return findPlayer(playerId).GetShares();
	}

	public int[] getPrices() {
		return StockPrices;
	}

	public Card[] getCards() {
		for(int i = 0; i < 5; i++){
			cards[i] = deck[i].cards.get(0);
		}
		return cards;
	}



	public void executeVotes() {

			for(int i = 0; i < Votes.length; i++){
				if(Votes[i] > 0){
					StockPrices[i] += cards[i].effect;
					deck[i].cards.remove(0);
					cards[i] = deck[i].cards.get(0);
				}
				else if(Votes[i] <0){
					deck[i].cards.remove(0);
					cards[i] = deck[i].cards.get(0);
				}
			}

			Votes = new int[5];
	}

	public String buy(int id, Stock s, int amount) {
		Player p = findPlayer(id);
		int StockPrice = getPrices()[s.ordinal()];
		if(p.GetCash() >= StockPrice*amount-(3*amount)){
			p.GetShares()[s.ordinal()] += amount;
			p.setCash(p.GetCash() - (StockPrice*amount)-(3*amount));
            return Integer.toString(id)+s.toString() + amount;
		}
		return "Not Enough Cash";

	}

	public String sell(int id, Stock s, int amount) {
		Player p = findPlayer(id);
		int StockPrice = getPrices()[s.ordinal()];
		if(p.GetShares()[s.ordinal()]>= amount) {
            p.GetShares()[s.ordinal()] -= amount;
            p.setCash(p.GetCash() + (StockPrice * amount));

            return Integer.toString(id) + s.toString() + amount;
        }
        return "Not Enough Stock";
	}

	public String vote(int id, Stock s,boolean yn ,boolean yes){
		if(yn) {
			if (yes) {
				Votes[s.ordinal()] += 1;
			}
		}
		if(yn == false){
			if(yes){
				Votes[s.ordinal()] -=1;
			}
		}
		return Integer.toString(id)+ s.toString() + String.valueOf(yes);
	}
	public void IsFinished(int id){
		Player p = findPlayer(id);
		p.finished = true;
	}
	public void Update() {

		if(players.size() > 0){
			boolean a = true;
		for (Player p : players) {
			if (p.finished == false) {
				a = false;
			}
		}
		if (a == true) {
			for(Player p : players) p.finished = false;
			executeVotes();
            if(Round == 5){
                ScoreCalc();
            }
            else {
                Round++;
            }
		}

	}
	}
	public void ScoreCalc(){

	    for(Player p : players){
	        for(int i = 0; i < 5; i++) {
                sell(p.playerID,Stock.values()[i],p.GetShares()[i]);
            }
        }



    }
}
