package game;

import java.awt.Color;
import java.awt.Graphics2D;

import utilities.Vector2D;
import static game.Constants.FRAME_WIDTH;
import static game.Constants.FRAME_HEIGHT;
public class Score extends GameObject {
	public static int sc = 0;
	public static int totalsc=0;
	
	String statlevel = "level : 1";
	String score = "The Score : "+sc;
	public Score() {
		super(new Vector2D(20,30),new Vector2D(0,0),20);
	}


	public void update() {
		score = "The Score : "+sc;
		
	}
	public void updatelevel(int level){
		statlevel = "level : "+level;
		
	}
	
	public void draw(Graphics2D g) {
		
		g.setColor(Color.green);
		g.setFont(g.getFont().deriveFont(30f));
		g.drawString(score, (int)position.x,(int)position.y);
		g.drawString(statlevel,FRAME_WIDTH-120 , 30);
		
	}


	
}
