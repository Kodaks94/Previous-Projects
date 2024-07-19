package game;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import javax.swing.JComponent;

public class View extends JComponent {
	// background colour
	public static final Color BG_COLOR = Color.black;
	Image im = Sprite.MILKYWAY; 
	AffineTransform bgTransf; 

	private Game game;

	public View(Game game) {
		this.game = game;
		double imWidth = im.getWidth(null); 
	    double imHeight = im.getHeight(null); 
	    double stretchx = (imWidth > Constants.FRAME_WIDTH? 1 : 
	                                Constants.FRAME_WIDTH/imWidth); 
	    double stretchy = (imHeight > Constants.FRAME_HEIGHT? 1 : 
	                                Constants.FRAME_HEIGHT/imHeight);
	    
	    bgTransf = new AffineTransform(); 
	    bgTransf.scale(stretchx, stretchy);
	    
	}

	@Override
	public void paintComponent(Graphics g0) {
		Graphics2D g = (Graphics2D) g0;
		// paint the background
		g.drawImage(im, bgTransf,null); 
		//g.setColor(BG_COLOR);
		//g.fillRect(0, 0, getWidth(), getHeight());
		
	
		
		synchronized(Game.class){
			for(int i = 0; i < game.objects.size();i++){
				game.objects.get(i).draw(g);
			}
			
		}
	}
		
	

	@Override
	public Dimension getPreferredSize() {
		return Constants.FRAME_SIZE;
	}
}