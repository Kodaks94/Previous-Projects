package BalanceBikeGame;

import com.sun.org.apache.bcel.internal.classfile.ConstantNameAndType;
import org.jbox2d.common.Vec2;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JComponent;

public class BasicView extends JComponent {
	/* Author: Michael Fairbank
	 * Creation Date: 2016-01-28
	 * Significant changes applied: Changed the paint component to work with my project
	 */
	// background colour
	public static final Color BG_COLOR = Color.BLACK;

	private PhysicsEngine game;

	public BasicView(PhysicsEngine game) {
		this.game = game;
	}
	
	@Override
	public void paintComponent(Graphics g0) {

		synchronized (this) {
			game = this.game;
		}
		Graphics2D g = (Graphics2D) g0;
		// paint the background
		g.setColor(BG_COLOR);
		g.fillRect(0, 0, getWidth(), getHeight());

		g.translate(-(int)CONSTANTS.convertWorldXtoScreenX(game.bikeparts.get(0).body.getPosition().x) + getWidth()/2,-(int) CONSTANTS.convertWorldYtoScreenY(game.bikeparts.get(0).body.getPosition().y) + getHeight()/2);
		g.setColor(Color.GREEN);

		g.drawString("SCORE= " + String.valueOf(game.gm.Score),+(int)CONSTANTS.convertWorldXtoScreenX(game.bikeparts.get(0).body.getPosition().x) + 800, getHeight()/4);
		g.drawString("Level= " + String.valueOf(game.gm.level),+(int)CONSTANTS.convertWorldXtoScreenX(game.bikeparts.get(0).body.getPosition().x) + 800,getHeight()/3.5f);
		game.terrainUtility.draw(g);
		for (Shapes s : game.shapes)
			s.draw(g);
		for (AnchoredBarrier b : game.barriers)
			b.draw(g);
		for (Bike p : game.bikeparts)
			p.draw(g);
		//game.water.draw(g);
		game.clouds.draw(g);

		game.sys.draw(g);





	}


	@Override
	public Dimension getPreferredSize() {
		return CONSTANTS.FRAME_SIZE;
	}
	
	public synchronized void updateGame(PhysicsEngine game) {
		this.game=game;
	}
}