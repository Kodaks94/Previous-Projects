package game;

import static game.Constants.DRAG;
import static game.Constants.DRAWING_SCALE;
import static game.Constants.DT;
import static game.Constants.FIRECOLOR;
import static game.Constants.FIREX;
import static game.Constants.FIREY;
import static game.Constants.FRAME_HEIGHT;
import static game.Constants.FRAME_WIDTH;
import static game.Constants.MAG_ACC;
import static game.Constants.OUTSHIPCOLOR;
import static game.Constants.OXP;
import static game.Constants.OYP;
import static game.Constants.SHIPCOLOR;
import static game.Constants.SHIP_RADIUS;
import static game.Constants.STEER_RATE;
import static game.Constants.XP;
import static game.Constants.YP;
import static game.Constants.FRAME_HEIGHT;
import static game.Constants.FRAME_WIDTH;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;

import utilities.SoundManager;
import utilities.Vector2D;
public class Ship extends Component {
    public Bullet bullet = null;
    public Bullet bullet2 = null;
    public boolean shooting = false;
    private Controller ctrl;
    public static int lives = 3;
    String livestat= "Lives: "+lives ; 
    public Ship(Controller ctrl) {
     super(new Vector2D(FRAME_WIDTH/2,FRAME_HEIGHT/2), new Vector2D(5,5),SHIP_RADIUS);
     this.ctrl = ctrl;
     direction = new Vector2D(5,5);
     
    
    }

    public void update(){
    	livestat ="Lives: "+ lives;
        direction.rotate(ctrl.action().turn * STEER_RATE * DT);
        this.velocity.addScaled(direction, MAG_ACC * DT * ctrl.action().thrust);
        this.velocity.subtract(DRAG, DRAG);
        this.velocity.SpeedCheck();
        this.position.addScaled(velocity, DT);
        this.position.wrap(FRAME_WIDTH, FRAME_HEIGHT);   
        ShootCheck();
        ThrustCheck();
        if(thrusting){
        	  SoundManager.startThrust();
        }
        else{
        	  SoundManager.stopThrust();
        }
        this.incrementvun();
        if(GameObject.powered){
			Bullet.R = 5;
		}
        else{
        	Bullet.R = 2;
        }
    }
   
    public void mkBullet(){
    	 Vector2D muzzleSpeed = new Vector2D(direction.x, direction.y);
    	muzzleSpeed = muzzleSpeed.normalise();
    	bullet = new Bullet(new Vector2D(position), new Vector2D(muzzleSpeed),1,this);
    	bullet.position.x =this.position.x;
    	bullet.position.y = this.position.y;
    	bullet.position.addScaled(direction, 5);
    	bullet.velocity.set(muzzleSpeed.addScaled(direction, MAG_ACC*DT));
    }
    public void reset(){
    	this.position.set(FRAME_WIDTH/2, FRAME_HEIGHT/2);
    }
    private void ThrustCheck(){
    	if(ctrl.action().thrust !=0){
    		this.thrusting = true;
    	}
    	else{
    		this.thrusting = false;
    	}
    }
    public void ShootCheck(){
    	if(ctrl.action().shoot){
        	mkBullet();
        	ctrl.action().shoot = false;
        	shooting = true;
        	
        }
        else{
        	shooting = false;
        }
    }
    public void draw(Graphics2D g) {
    	  AffineTransform at = g.getTransform();
    	  g.translate(this.position.x, this.position.y);
    	  double rot = direction.angle() + Math.PI / 2;
    	  g.rotate(rot); 
    	 g.scale(DRAWING_SCALE, DRAWING_SCALE);
    	  if (thrusting) {
      		g.setColor(FIRECOLOR);
      		g.fillPolygon(FIREX, FIREY, FIREX.length);
      		
      	  }
    	  g.setColor(OUTSHIPCOLOR);
    	  g.fillPolygon(OXP, OYP, OXP.length);
    	  g.setColor(SHIPCOLOR);
    	  g.fillPolygon(XP, YP, XP.length);
    	  
    	
    
    	  g.setTransform(at);
    	  g.setColor(Color.GREEN);
    	  g.setFont(g.getFont().deriveFont(30f));
    	  g.drawString(livestat, FRAME_WIDTH-130, FRAME_HEIGHT-50);
    	  
    	}
    	
}