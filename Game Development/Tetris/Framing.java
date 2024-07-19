package myTetris;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Toolkit;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class Framing extends JFrame { 
	JLabel status;
	int w = 10, h = 20;
    static int size = 25;
    JButton ButReset;
  
	public Framing(){ // adds all the components of the frame such as reset button and gamepart section and calls the controls  sets the size of the frame and sets the background color of the grame 
		
		JPanel SR = new JPanel();
		status = new JLabel(" 0");
		ButReset = new JButton("Reset");
		GamePart game = new GamePart(this);
		BlockMovements XBM = new BlockMovements(game);  
		Controls control = new Controls(game,XBM,this);
		ButReset.addActionListener(control);
		SR.add(status);
		SR.add(ButReset);
		add(SR,BorderLayout.PAGE_END);
		
		setSize(w*size,h*size);
		setTitle("Mahrad Pisheh Var 1404537");
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize(); // sets in middle of the screen 
		this.setLocation(dim.width/2-this.getSize().width/2, dim.height/2-this.getSize().height/2);
		game.setBackground(Color.BLACK);
		
		add(game);
		
	}
	public JLabel ReturnStatus(){
		return status;
	}
}
