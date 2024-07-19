package myTetris;

import static java.awt.Color.black;//importing the Colors needed
import static java.awt.Color.blue;
import static java.awt.Color.cyan;
import static java.awt.Color.green;
import static java.awt.Color.magenta;
import static java.awt.Color.pink;
import static java.awt.Color.red;
import static java.awt.Color.yellow;

import java.awt.Color;		
import java.awt.Dimension;	
import java.awt.Graphics;	
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseListener;
import java.util.Collections;
import java.util.Random;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.Timer;

import myTetris.Shapes.TetrisShapes;

public class GamePart extends JPanel{		//part of the program where the main game part of the program is held 
	DrawingBlocks draw = new DrawingBlocks(this);
	Scores Sc = new Scores(this);
	BlockMovements BM= new 	BlockMovements(this);
	
	final int w = 10,h= 20; // setting the height and lenghts of the game part of the frame
	Timer timer;
	boolean FallComplete= false, started = false, paused = false;
	int score = 0; // score variable   
	int currentX = 0, currentY = 0; // Current Position of the Shape assigned to zero
	JLabel status;
	Shapes currentPiece;
	TetrisShapes[] game;	//declaring the TetrisShapes array with name game
       int Speed = 400;  //speed of the game which will be decreasing when the player scores
      Controls control;  
	public GamePart(Framing app){ // constructor that will initialise the timer and keeps the current status and clears the frame from blocks if there are there to begin the new game 
								// this constructor will also adds Mouse Listener and Key Listener from other class to keep the track of button inputs
	
		setFocusable(true);
		currentPiece = new Shapes();
		Controls control = new Controls(this,BM,app);
		timer = new Timer(Speed, control);
		timer.start();
		status = app.ReturnStatus();
		game = new TetrisShapes[w*h];
		draw.clear();
		Mouse m = new Mouse(this,BM);
		Keyboard k = new Keyboard(this,BM);
		addMouseListener(m);
		addKeyListener(k);
		
	}
	TetrisShapes ReturnEmpty(){ // return the NoShape element of TetrisShapes
		return TetrisShapes.NONE;
	}
	TetrisShapes ShapeLocation(int x, int y) { //Setting the Shape Location with parameters x and y 
		return game[(y * w) + x];
		}
	TetrisShapes ShapeClear(int x, int y){ // this is used further down in other classes if the line is full of blocks it will empty the row but for single use it will clear the shape chosen with coordination of x and y
		return game[(y*w) +x] = ReturnEmpty();
	}
	public void Start(){ // starts the timer and initialize the game 
		if(paused){
			return;
		}
	
		started = true;
		FallComplete = false;
		score = 0;
		draw.clear();
		timer.start();
	}
	
	 public void paint(Graphics g) // paint will paint the shapes according to the 
	    { 
		 Dimension len = getSize();
	      super.paint(g);
	        
	        
	      	int topmax = (int) len.getHeight() - h * SgetHeight(); // gets the highest point of the shape in the frame coordiantes
	        for (int x = 0; x < h; ++x) {
	            for (int y = 0; y < w; ++y) {
	                TetrisShapes shape = ShapeLocation(y, h - x - 1); 
	                if (shape != TetrisShapes.NONE)		//if the shape is not none 
	                    draw.BlockMaker( 0 + y * SgetWidth(),topmax + x * SgetHeight(),g,shape);	
	            }
	        }

	        if (currentPiece.getShape() != ReturnEmpty()) { // if the chosen shape is not none makes a new one 
	           paintingNew(topmax,g);
	            }
	        }
	    int SgetWidth() { //gets the Width of the shape
			return (int) getSize().getWidth() / w;
			}
		int SgetHeight() { //gets the height of the shape
			return (int) getSize().getHeight() / h; 
			}

	 public void pieceDropped(){ // this function will check when the piece is dropped and makes a new boundary for shape that wont collide with other shape and it will the the scoring() function from Score class
		 
		 for (int i = 0; i < 4; ++i) {
	            int x = currentPiece.getx(i)+currentX;
	            int y =currentY-currentPiece.gety(i);
	            game[(y * w) + x] = currentPiece.getShape();
	        }
		 Sc.scoring();
		 if(!FallComplete){
			 getnewPiece();
		 }
	 }
	public void Speed(int speed){
		timer.setDelay(speed);
	}
	 public void getnewPiece() // generates the new shape 
	    {
		 	
	        currentPiece.randomShape(); 
	        currentY =h-1+currentPiece.miniY();
	        currentX =(w/2)+1;
	        if (!BM.MoveTetris(currentX, currentY,currentPiece)) { // if the block is out the boundary as its generated  makes a new shape with empty shape and stops the timer and it will give the game over message with option for the user to save
	            currentPiece.setting(ReturnEmpty());
	            timer.stop();
	            started = false;
	            String scores = status.getText();
	            status.setText("Game Over");
	            String[] buttons = {"Save","Cancel"};
	            String Player = null;
	            int rc = JOptionPane.showOptionDialog(null, "Would you like to save?", "Save", JOptionPane.WARNING_MESSAGE, 0, null, buttons, buttons[1]);
	            if(rc == 0){
	            	Player = JOptionPane.showInputDialog("enter Name");
	            	Save s = new Save();
	            	s.saving(Player,scores);
	            	s.Loading();
	            }
	        }
	    }
	
	 public void paintingNew(int top, Graphics g){ // generates the shape from the position that top integer is coordinated assigns the grid position of the shape
		 
		 for (int x = 0; x < 4; ++x) {
             int i = currentX + currentPiece.getx(x);
             int j = currentY - currentPiece.gety(x);
             draw.BlockMaker( 0 + i * SgetWidth(),top + (h - j - 1) * SgetHeight(), g,currentPiece.getShape());
	 }
	 }
	

}
