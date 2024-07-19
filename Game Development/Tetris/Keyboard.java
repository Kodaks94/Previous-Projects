package myTetris;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

public class Keyboard extends KeyAdapter { // this class will keep the track of the keyboard button 
	GamePart theApp;
	BlockMovements BM;
	public Keyboard(GamePart app,BlockMovements bm){
		theApp = app;
		BM = bm;
	}
	public void keyPressed(KeyEvent e){ 
		if(e.getKeyCode() == KeyEvent.VK_SHIFT){ // if shift is clicked it will drop the blocks down to where the boundary is and it will call the pieceDropped() function from GamePart class which will get the new piece ready 
			int temp = theApp.currentY;
			while(temp > 0){
				if(!BM.MoveTetris(theApp.currentX, temp-1, theApp.currentPiece)){
					break;
				}
				--temp;
			}
			theApp.pieceDropped();
		
		}
	}

}
