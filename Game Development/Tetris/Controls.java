package myTetris;

import java.awt.event.*;


public class Controls implements ActionListener{ // this class will determine if the fall is complete if it is makes a new piece also it will execute the reset button which will clears the line by line of the grid 
	GamePart theApp;
	Framing frame;
	BlockMovements BM;
	Controls(GamePart app,BlockMovements xBM,Framing fr){
		theApp = app;
		frame =fr;
		BM = xBM;
	}
	public void actionPerformed(ActionEvent e){
		
		if(theApp.FallComplete){
			theApp.FallComplete = false;
			theApp.getnewPiece();
		}
		
		
		
		if(e.getSource() == frame.ButReset ){
			for(int i = theApp.h-1 ; i >=0; --i){
				for(int j = 0; j < theApp.w; ++j){
					theApp.ShapeClear(j, i);
				}
			}
			theApp.started = true;
			theApp.timer.start();
			theApp.status.setText(" 0");
		}
		
		else BM.goingdownOne(); // if none makes the shape go down by 1 
	}
}
