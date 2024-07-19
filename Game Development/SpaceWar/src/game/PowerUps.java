package game;

import static game.Constants.FRAME_HEIGHT;
import static game.Constants.FRAME_WIDTH;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import utilities.Vector2D;

public class PowerUps extends GameObject {
	
	public static ArrayList<PowerUps> powers = new ArrayList<PowerUps>();
	public static HashMap<GameObject,Integer> used = new HashMap<GameObject,Integer>();
	public PowerUps(Vector2D position, Vector2D velocity, double radius,int power) {
		super(position, velocity, radius);
		this.power = power;
		powers.add(this);
	}

	@Override
	public void update() {
		updatetime();
		if(this.updating == 800){
			this.dead = true;
		}
		
	}
	public void addPower(PowerUps p){
		powers.add(p);
	}
	@Override
	public void draw(Graphics2D g) {
		if(power == 1){
			g.setColor(Color.GREEN);
			
		}
		
		else if(power == 2){
			g.setColor(Color.RED);
		}
		
		g.drawOval((int)(position.x), (int) position.y, (int)radius*2,(int) radius*2);
		g.fillOval((int)(position.x+radius/2), (int)(position.y+radius/2), (int)radius,(int) radius);
		
	}
	public static void powerdrop(GameObject object){
		
		used.put(object,1);
		if(used.get(object) == 1){
		Random rand = new Random();
		
		int b = rand.nextInt(2)+1;
		
		int  n = rand.nextInt(50) + 1;
		if(n == 1){
			double x = Math.random()*FRAME_WIDTH;
	        double y = Math.random()*FRAME_HEIGHT;
	        System.out.println(b);
			new PowerUps(new Vector2D(x,y),new Vector2D(0,0),20,b);
	    	}
			
		}
		
	}

}
