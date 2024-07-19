package game;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Image;

public class Constants {
	public static final int FRAME_HEIGHT = 565;
	public static final int FRAME_WIDTH = 1023;
	public static final Dimension FRAME_SIZE = new Dimension(Constants.FRAME_WIDTH,Constants.FRAME_HEIGHT);
	public static final int DELAY = 10;
	public static final double DT = DELAY/1000.0 ;
	 // rotation velocity in radians per second 
    public static final double STEER_RATE =  Math.PI;  
    // acceleration when thrust is applied
    public static final double MAG_ACC = 200; 
    // constant speed loss factor 
    public static final double DRAG = 0.01;
    public static final Color COLOR = Color.cyan;
    public static final Color SHIPCOLOR = new Color(0, 0, 102);
    public static final Color OUTSHIPCOLOR = new Color(0, 102, 0);
    public static final Color FIRECOLOR= new Color(102, 153, 255);
    public static final double DRAWING_SCALE = 10;
    public static final int OXP[] = {-2,0,2,0};
    public static final int OYP[] = {4,2,4,-2};
    public static final int XP[] = {-1,0,1,0};
    public static final int YP[] = {2,1,2,-1};
    public static final int THRUSTXP[] =  {-1,0,1,0};
    public static final int THRUSTYP[] ={2,1,2,-1}; 
    public static final int FIREX[] ={-1,-1,0,1,1};
    public static final int FIREY[] ={2,4,3,4,2};

    
    public static final int SHIP_RADIUS = 20;
    // direction in which the nose of the ship is pointing 
    // this will be the direction in which thrust is applied 
    // it is a unit vector representing the angle by which the ship has rotated 
   public static final int ASTEROID_RADIUS = 10;
   public static final int ASTEROID_MAX_SPEED = 100;
   public static final int CANNON_RADIUS = 20;
   public static final int XC[] = {-1,1,1,-1};
   public static final int YC[] = {4,4,0,0};
public static final Image MILKYWAY = null;
   
}
