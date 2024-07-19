package game;
import static game.Constants.COLOR;
import static game.Constants.MAG_ACC;
import static game.Constants.CANNON_RADIUS;
import static game.Constants.DRAWING_SCALE;
import static game.Constants.DT;
import static game.Constants.FRAME_HEIGHT;
import static game.Constants.FRAME_WIDTH;
import static game.Constants.THRUSTXP;
import static game.Constants.THRUSTYP;
import static game.Constants.XC;
import static game.Constants.YC;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import utilities.Vector2D;
import static game.Constants.STEER_RATE;
public class Cannon extends Component{
	public static ArrayList<Cannon> cannons = new ArrayList<Cannon>();
	public Bullet bullet = null;
	private Ship ship;
	public boolean shooting;
	public Cannon(Vector2D position, Vector2D velocity, double radius) {
		super(position, velocity, radius);
		this.SetVisionRadius(250);
		while(SettingCannons() == false);
		this.setDirection(position);
		this.addCannon(this);
	}
	
	public boolean  SettingCannons(){
		for(Cannon a : cannons){
			for (Cannon b: cannons){
				if(!a.equals(b)){
				if(Math.abs(a.position.x-b.position.x) <= 100 && Math.abs(a.position.y - b.position.y) <= 100){
					cannons.remove(a);
					RandomlyGeneratedCannons();
					return false;
				}
			}
			}
		}
		return true;
	}
	public void mkBullet(){
		
	
		Vector2D muzzleSpeed = new Vector2D(direction.x, direction.y);
    	muzzleSpeed = muzzleSpeed.normalise();
    	bullet = new Bullet(new Vector2D(position), new Vector2D(muzzleSpeed),2,this);
    	bullet.position.x =this.position.x;
    	bullet.position.y = this.position.y;
    	bullet.position.addScaled(direction, 5);
    	bullet.velocity.set(muzzleSpeed.addScaled(direction, MAG_ACC*DT));
		
		
	}
	
	private void setDirection(Vector2D pos){
		if(pos.x ==0){
			direction.set(0,-1);
		}
		else if(pos.x == FRAME_WIDTH){
			direction.set(0,1);
		}
		else if(pos.y == 0){
			direction.set(-1, 0);
		}
		else if(pos.y == FRAME_HEIGHT){
			direction.set(1, 0);
		}
		
	}
	public void updateShip(Ship ship){
		this.ship = ship;
	}
	public static Cannon RandomlyGeneratedCannons(){
		double x=0,y=0,vx =0,vy=0;
		Random r = new Random();
		int sideChoice = r.nextInt(5-1)+1;
		
		if(sideChoice == 1){
			y = Math.random()*FRAME_HEIGHT;
			
		}
		else if(sideChoice == 2)x= Math.random()*FRAME_WIDTH;
		else if (sideChoice ==3 ){
			x = FRAME_WIDTH;
			y = Math.random()*FRAME_HEIGHT;
		}
		else if(sideChoice == 4){
			x = Math.random()*FRAME_WIDTH;
			y = FRAME_HEIGHT;
		}
		return new Cannon(new Vector2D(x,y),new Vector2D(vx,vy),CANNON_RADIUS);
	}

	private void addCannon(Cannon b){
		cannons.add(b);
		
	}
	public void ShootCheck(){
		if(this.AIM(ship)){
			if(bullet == null){
			this.mkBullet();
			}
			else if(bullet.dead){
				this.mkBullet();
			}
		}
	}
	@Override
	public void update() {
		ShootCheck();
		this.updatemove();
		direction.rotate(turn * STEER_RATE * DT);
		this.velocity.addScaled(direction, MAG_ACC*DT*thrust);
		this.position.addScaled(velocity, DT);
		
		this.incrementvun();
	}

	@Override
	public void draw(Graphics2D g) {
		AffineTransform at = g.getTransform();
	  	  g.translate(this.position.x, this.position.y);
	  	   double rot = direction.angle() ;
	  	//  double rot = new Vector2D(FRAME_WIDTH/2,FRAME_HEIGHT/2).subtract(position).angle();
	  	  g.rotate(rot + 1.5*Math.PI);
	  	  g.scale(DRAWING_SCALE, DRAWING_SCALE);
	  	  g.setColor(Color.ORANGE);
	  	  g.fillPolygon(XC, YC, XC.length);
	  	  g.setTransform(at);
	  	  
	  	  g.fillOval((int)this.position.x - (int)this.radius, (int) this.position.y - (int)this.radius, 2 * (int)this.radius, 2 * (int)this.radius);
	  	  
	  	  
	}

}
