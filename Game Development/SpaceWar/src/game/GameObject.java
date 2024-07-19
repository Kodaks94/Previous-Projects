package game;

import java.awt.Graphics2D;
import java.util.HashMap;

import utilities.Vector2D;
public abstract class GameObject {
	public Vector2D position;
	public Vector2D velocity;
	public boolean dead;
	public double radius;
	public int vun;
	public int power;  // health // radius
	public int mutlevel;
	public static boolean powered = false;
	public double VisionRadius;
	public HashMap<String, Boolean> Move;
	public boolean thrusting;
	public int BulletReferernce = 0; // 1 Ship bullet 2 Cannon bullet
	int thrust;
	int turn;
	int updating;
	static boolean used = false;
	public Vector2D direction;
	public GameObject(Vector2D position, Vector2D velocity, double radius){
		this.position = position;
		this.velocity = velocity;
		this.radius = radius;
		this.dead = false;
		thrust = 0;
		turn = 0;
		this.thrusting = false;
		 Move = new HashMap();
		vun = 0;
		direction = new Vector2D(0,0);
	}
 
   public boolean overlap(GameObject other){
	 	double distance = this.position.dist(other.position);
	 	return distance < (this.radius+other.radius);
	}
   public boolean CollisionHandling(GameObject other){
	   if (this.getClass() != other.getClass() && this.overlap(other)&&this.vun == 200){
		   this.hit();
		   other.hit();
		   return true;
	   }
	   return false;
	   
	 
   }
   public boolean CollisionBoolean(GameObject other){
	   if (this.getClass() != other.getClass() && this.overlap(other)&&this.vun == 200){
		   return true;
	   }
	   return false;
   }
   
   public void SetVisionRadius(double x){
	   VisionRadius = x;
   }
	public  void hit(){
		this.dead = true;
	}
	public void incrementvun(){
		if(vun != 200)vun++;
	}
	public void updatetime(){
		updating++;
	}
	public abstract void update();
	public abstract void draw (Graphics2D g);
	 
}
