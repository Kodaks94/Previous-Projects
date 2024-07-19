package game;

import java.awt.Graphics2D;

import utilities.Vector2D;

public abstract class Component extends GameObject {
	
	static Vector2D anglerotate = new Vector2D(0,0);
	public static int AmmoCount = 0;
	public Component(Vector2D position, Vector2D velocity, double radius) {
		super(position, velocity, radius);
		Move.put("Left", false);
		Move.put("Right", false);
		Move.put("Thrust", false);
	}
	
	public int counter = 0; 
	public abstract void update();
	public abstract void mkBullet();
	public abstract void ShootCheck();
	public abstract void draw(Graphics2D g);
	public boolean AIM(Component other){
		double distance  = other.position.dist(this.position);
		if(distance < other.radius+ this.VisionRadius){
			Vector2D apparentDirection = new Vector2D(other.position).subtract(position);
			double angle = direction.angle(apparentDirection); 
			counter++; 	
			if(angle < 0){
				moveLeft();
				return true;
			}
			else if (angle > 0){
				
				moveRight();
				return true;
			}
			
			else{
				return false;
			}
		}
		else {
			Stopturn();
			return false;
		}
	}
	private void moveLeft(){
		this.Move.put("Right", false);
		this.Move.put("Left", true);
		
	}
	private void moveRight(){
		this.Move.put("Left", false);
		this.Move.put("Right", true);
	}
	private void moveForward(){
		this.Move.put("Thrust", true);
		thrusting = true;
	}
	private void moveStop(){
		this.Move.put("Thrust", false);
		thrusting  = false;
	}
	private void Stopturn(){
		this.Move.put("Left", false);
		this.Move.put("Right", false);
	}
	public
	void updatemove(){
		if(Move.get("Left") == true){
			turn = -1;
			
		}
		else if(Move.get("Right") == true){
			turn = 1;	
		}
		else if(Move.get("Left")==false && Move.get("Right") == false){
			turn = 0;
			
		}
		if(Move.get("Thrust") == true){
			thrust = 1;
			
		}
		else{
			thrust = 0;
			
		}
	}
	
	
}
