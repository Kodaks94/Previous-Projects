package myTetris;

import static java.awt.Color.black;
import static java.awt.Color.blue;
import static java.awt.Color.cyan;
import static java.awt.Color.green;
import static java.awt.Color.magenta;
import static java.awt.Color.pink;
import static java.awt.Color.red;
import static java.awt.Color.yellow;

import java.awt.Color;
import java.awt.Graphics;
import java.util.List;
import java.util.Random;

import myTetris.Shapes.TetrisShapes;

public class DrawingBlocks { // draws the block which each shape will have the same color 
	GamePart theApp;
	  private Color[] colorlist =   {black,green, blue, red,
            yellow, magenta, pink, cyan};
	
	
	DrawingBlocks(GamePart app){
		theApp = app;
		
	}
	public void BlockMaker(int x, int y , Graphics g , TetrisShapes shape){ // draws the shape given to the method with coordination 
		Color c = colorlist[shape.ordinal()];
		g.setColor(c);
		g.fillRect(x+1, y+1, theApp.SgetWidth(), theApp.SgetHeight());
			
	}
	public void clear() // clears the full grid of shapes
    {
		 int FullGrid = theApp.h*theApp.w;
        for (int i = 0; i < FullGrid; ++i)
            theApp.game[i] = theApp.ReturnEmpty();
    }
	
}
