package myTetris;

import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class Mouse extends MouseAdapter { // this will keep track of mouse buttons if its clicked
	GamePart theApp;
	BlockMovements BM;
	Mouse(GamePart app,BlockMovements xBM){
		theApp = app;
		BM = xBM;
		
	}
	public void mousePressed(MouseEvent e){
		if(e.getButton() == MouseEvent.BUTTON3){ //right click will move the shape to the right 
			BM.MoveTetris( theApp.currentX+1, theApp.currentY,theApp.currentPiece);
		
		}
		if(e.getButton() == MouseEvent.BUTTON1){ // left click will move the shape to the left
			BM.MoveTetris( theApp.currentX-1, theApp.currentY,theApp.currentPiece);
		}
		if(e.getButton() == MouseEvent.BUTTON2){ // middle / scroll button click will rotate them to the right 
			BM.MoveTetris(theApp.currentX, theApp.currentY,theApp.currentPiece.rotateRight());
		}
		
	}
	}

