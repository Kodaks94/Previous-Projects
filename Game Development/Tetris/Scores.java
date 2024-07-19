package myTetris;

import myTetris.Shapes.TetrisShapes;

public class Scores { //keeps the scores and Adds up the scores if the line is full and clears it the line if its filled with blocks 
	GamePart theApp;
	private int scored = 0;

	Scores(GamePart app){
		theApp = app;
	}
	public void scoring()  //  this checks the line by line of the grids if the line is filled with blocks it increments the score and it will move down the blocks 
    {
       scored = 0;
        for (int x = theApp.h - 1; x >= 0; --x) {
            
            //scored
            if (isFull(x)) {// this loop is used if there are two or more rows of blocks are filled
                ++scored;
                for (int z = x; z < theApp.h - 1; ++z) {
                    for (int y = 0; y < theApp.w; ++y)
                         movedown(x,y,z);
                }
            }
        }
        if (scored > 0) {
            plusScore(scored);
            theApp.Speed(theApp.Speed);
        }
    }
	
	public void movedown(int x, int y , int z){ // moves one x variable down if the lines all cleared by incrementing the x 
		theApp.game[(z * theApp.w) + y] = theApp.ShapeLocation(y, z + 1);
	}
	public void plusScore(int S){ // increases the score and decreases the timer speed which increase the movements speed of the blocks 
		theApp.score += S*10;
		if(theApp.score%10== 0){
			theApp.Speed-=10;
		}
         theApp.FallComplete = true;
         String sc = String.valueOf(theApp.score);
         theApp.status.setText(sc);
        theApp.currentPiece.setting(theApp.ReturnEmpty());
        theApp.repaint();
	}
	public Boolean isFull(int x){ //checks if the line of the grid is filled with blocks 
		Boolean full = null;
            full = true;
            for (int y = 0; y < theApp.w; ++y) {
                if (theApp.ShapeLocation(y, x) == theApp.ReturnEmpty()) {
                    full = false;
                    break;
                }
            }
	
		return full;
	
}
}
