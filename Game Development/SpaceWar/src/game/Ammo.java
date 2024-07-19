package game;

import java.awt.Color;
import java.awt.Graphics2D;
import static game.Constants.FRAME_HEIGHT;
import utilities.Vector2D;

public class Ammo extends Component{
	private String stat = "AMMO";
	public Ammo() {
		super(new Vector2D(10,FRAME_HEIGHT-10), new Vector2D(0,0), 20);
	}

	
	public void update() {
		if(AmmoCount == 5)stat = "RELOADING";
		else stat = "AMMO";
		
	}


	public void draw(Graphics2D g) {
		g.setColor(Color.green);
		for(int i = 1; i <= (5-AmmoCount); i++){
			g.fillRect(((int)this.position.x+20)*i, (int)this.position.y - 40, 20, 20);
		}
		g.setColor(Color.green);
		g.drawLine((int) position.x +10, (int) position.y-50, (int)position.x+10, (int) position.y-10);
		g.drawLine((int) position.x +10, (int) position.y-10, (int)position.x *5, (int) position.y-10);
		g.drawString(stat,(int)position.x +13, (int) position.y-53);
	}


	@Override
	public void mkBullet() {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void ShootCheck() {
		// TODO Auto-generated method stub
		
	}
	

}
