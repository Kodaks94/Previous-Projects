package game;

import static game.Constants.ASTEROID_MAX_SPEED;
import static game.Constants.ASTEROID_RADIUS;
import static game.Constants.DT;
import static game.Constants.FRAME_HEIGHT;
import static game.Constants.FRAME_WIDTH;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.HashMap;

import utilities.Vector2D;


public class Asteroid extends GameObject{
    public static  ArrayList<Asteroid> SpawnedAsteroids= new ArrayList<Asteroid>();
    public static HashMap<GameObject,Integer> usedAsteroid = new HashMap<GameObject,Integer>();
    Image im = Sprite.ASTEROID;
    public static int Multiply;
    public Asteroid(double x, double y, double vx, double vy, int RADIUS) {
    	super( new Vector2D(x, y),new Vector2D(vx, vy),RADIUS);
    	addAsteroid(this);
    	this.mutlevel = 1;
    	
    }
   
    public  void addAsteroid(Asteroid b){
    	
    	SpawnedAsteroids.add(b);
    	
    }
    public static Asteroid makeRandomAsteroid() {
        double x = Math.random()*FRAME_WIDTH;
        double y = Math.random()*FRAME_HEIGHT;
        double vx = Math.random()*ASTEROID_MAX_SPEED*2-ASTEROID_MAX_SPEED;
        double vy = Math.random()*ASTEROID_MAX_SPEED*2-ASTEROID_MAX_SPEED;
        return new Asteroid(x, y, vx, vy,ASTEROID_RADIUS); //temp
    }

    public void update() {
        position.addScaled(this.velocity, DT);
        position.wrap(FRAME_WIDTH, FRAME_HEIGHT);
        this.incrementvun();
    }

    public void draw(Graphics2D g) {
        g.drawImage(im, (int)this.position.x - (int)this.radius, (int) this.position.y - (int)this.radius,2 * (int)this.radius, 2 * (int)this.radius, null);
    }
    public static void mutation(GameObject as){
    	
    	if(usedAsteroid.containsKey(as)){
    		usedAsteroid.put(as,usedAsteroid.get(as)+1);
    	}
    	else{
    		usedAsteroid.put(as, 1);
    	}
    	
    	if(as.mutlevel != Multiply){
    		if(usedAsteroid.get(as) <= Multiply){
    			
    		double x = as.position.x;
    		double y = as.position.y;
    		double vx = Math.random()*ASTEROID_MAX_SPEED*2-ASTEROID_MAX_SPEED;
            double vy = Math.random()*ASTEROID_MAX_SPEED*2-ASTEROID_MAX_SPEED;
            Asteroid b = new Asteroid(x,y,vx,vy,(int)as.radius/2);
           
            b.mutlevel = as.mutlevel+1;
    	}
    }
    }
}
