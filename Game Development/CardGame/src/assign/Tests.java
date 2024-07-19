package assign;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;

public class Tests {
	


	public Game game;

	// sample data
	public static Deck[] sampleDecks() {
		return new Deck[] { 
				new Deck(Stock.Apple, -20, 5, 10, -5, -10, 20), 
				new Deck(Stock.BP, 20, 5, 10, -5, -10, -20),
				new Deck(Stock.Cisco, 20, -5, 10, -20, -10, 5), 
				new Deck(Stock.Dell, 5, -20, 10, 20, -10, -5),
				new Deck(Stock.Ericsson, -10, 10, 20, 5, -20, -5) 
		};
	}

	public static int[][] sampleShares() {
		return new int[][] { 
			{ 3, 0, 1, 4, 2 }, 
			{ 2, 2, 5, 0, 1 }, 
			{ 4, 1, 0, 1, 4 } 
		};
	}
	public Tests(){

	}
	@Before
	public void setup() {
		game = new Game();
		Game.deck = sampleDecks();
		game.addPlayer();
		game.addPlayer();
		game.addPlayer();
		int i = 0;

		for(Player a : game.players){
			a.shares = sampleShares()[i];
			i++;
		}
	}
	@After
	public void setupafter(){
		game.players = new ArrayList<>();
		game.numofplayers = 0;
	}
	@Test
	public void tradeP0() {
		game.sell(1, Stock.Apple, 3);
		game.buy(1, Stock.Cisco, 6);
		Assert.assertArrayEquals(game.getShares(1), new int[] { 0, 0, 7, 4, 2 });
		Assert.assertEquals(Integer.parseInt(game.getCash(1)), 182);
	}

	@Test
	public void tradeP1() {
		game.buy(2, Stock.BP, 4);
		Assert.assertArrayEquals(game.getShares(2), new int[] { 2, 6, 5, 0, 1 });
		Assert.assertEquals(Integer.parseInt(game.getCash(2)), 88);
	}

	@Test
	public void tradeP2() {
		game.sell(3, Stock.Ericsson, 4);
		game.sell(3, Stock.Apple, 4);
		Assert.assertArrayEquals(game.getShares(3), new int[] { 0, 1, 0, 1, 0 });
		Assert.assertEquals(Integer.parseInt(game.getCash(3)), 1300);
	}

	@Test
	public void vote() {
		game.vote(1, Stock.Cisco, true,true);
		game.vote(1, Stock.Ericsson, false,true);
		game.vote(2, Stock.BP, true,true);
		game.vote(2, Stock.Cisco, true,true);
		game.vote(3, Stock.Apple, true,true);
		game.vote(3, Stock.BP, false,true);
		game.getCards();
		game.executeVotes();
		Assert.assertArrayEquals(game.getPrices(), new int[] { 80, 100, 120, 100, 100 });
		Card[] cards = new Card[] { new Card(5), new Card(20), new Card(-5), new Card(5), new Card(10) };
		Assert.assertArrayEquals(game.getCards(), cards);
	}

}
