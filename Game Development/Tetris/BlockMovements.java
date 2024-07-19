package myTetris;

import myTetris.Shapes.TetrisShapes;

public class BlockMovements { // this class will check if any movements is available for the shape 
	GamePart theApp;
	BlockMovements(GamePart app){
		theApp = app;
	}
	 
	public boolean MoveTetris(int nX, int nY, Shapes newShape) // checks the coordination of the shape and if its out of boundary or colliding with others it will give false if not it will proceed on changing the location of the shape
    {
		int i = 0;
		while(i < 4)
        {
            int y = nY - newShape.gety(i);
            int x = nX + newShape.getx(i);
            if (isOutOfBoundary(x,y))return false;
            if(isColliding(x,y))return false;
        ++i;
		}

      ChangeLocation(nX, nY,newShape);
      
      return true;
    }
	public void goingdownOne(){ // makes the shape go down by 1 
		 if(!MoveTetris(theApp.currentX, theApp.currentY -1,theApp.currentPiece)){
			 theApp.pieceDropped();
		 }
	 }
	public Boolean isOutOfBoundary(int x , int y){ // checks if its out of the boundary 
		if(x>= theApp.w|| y >= theApp.h ||x <0 || y<0){
			return true;
		}
		return false;
		
	}
	public Boolean isColliding(int x, int y){ // checks if its colliding with other shapes
		
		if(theApp.ShapeLocation(x,y)!= TetrisShapes.NONE){
			return true;
			
		}
		return false;
	}
	public void  ChangeLocation(int x, int y , Shapes shape){ // changes the location of the pieces which will get the current coordination and changes with given coordianation 
		theApp.currentX = x;
		theApp.currentPiece = shape;
		theApp.currentY = y;
		theApp.repaint();
		
	}
}
