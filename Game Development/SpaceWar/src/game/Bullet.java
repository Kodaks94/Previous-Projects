package game;

import static game.Constants.DRAG;
import static game.Constants.DT;
import static game.Constants.FRAME_HEIGHT;
import static game.Constants.FRAME_WIDTH;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

import utilities.SoundManager;
import utilities.Vector2D;

public class Bullet extends Component {
	public static  ArrayList<Bullet> bl = new ArrayList<Bullet>();
	public static ArrayList<Bullet> cbl = new ArrayList<Bullet>();
	public static int R = 2;
	Cannon can;
	Ship ship;
	public Bullet(Vector2D position, Vector2D velocity, int BulletReference, Cannon can){
		super(position, velocity,R);
		this.can = can;
		this.BulletReferernce = BulletReference;
		addBullet(this);
		
	}
	public Bullet(Vector2D position, Vector2D velocity, int BulletReference, Ship ship){
		super(position, velocity,R);
		this.ship = ship;
		this.BulletReferernce = BulletReference;
		addBullet(this);
		
		
	}

	private void addBullet(Bullet b){
		
		int  count  = 0;
		if(this.BulletReferernce == 1){
		if(bl.size() >=5){
		for (Bullet a : bl){
			
			if (a.dead) count++;
		}
	
		if (count == bl.size()) {
			bl.clear();
			GameObject.powered = false;
			bl.add(b);
			SoundManager.fire();
		}
		}
		else{
			bl.add(b);
			SoundManager.fire();
		}
		}
		if(this.BulletReferernce == 2){
			cbl.add(b);
			SoundManager.fire();
			
			}
		
	}

	@Override
	public void update() {
		AmmoCount = bl.size();
		position.addScaled(this.velocity, DT);
		if(this.position.x > FRAME_WIDTH || this.position.y > FRAME_HEIGHT || this.position.x <0 || this.position.y < 0){
			this.dead = true;
		}
		if(this.BulletReferernce == 1){
		if(this.position.dist(ship.position) > 600){
			this.dead = true;
		}
		}
		if(this.BulletReferernce == 2){
			if(this.position.dist(can.position) > 400){
				this.dead = true;
			}
		}
		
		this.incrementvun();
		
	}

	@Override
	public void draw(Graphics2D g) {
		g.setColor(Color.red);
		g.fillOval((int)this.position.x-R,(int)this.position.y-R,2*R,2*R);
	}
	@Override
	public void mkBullet() {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void ShootCheck() {
		
	}
	
}
