package myTetris;
import java.util.Random;
import java.math.*;
public class Shapes {
	enum TetrisShapes {NONE,Z,S,I,T,SQ,L,ML}; // making an enum of all the shapes will be included 
	private int[][][] Graph;	//graph of coordination  for defining the shape
	private int cordination[][]; //coordination of the shape 
	private TetrisShapes TetrisPiece;
	public Shapes(){		//initialize the coordination all to zero  each shape will have 2 elements of 4 different coordination 
		cordination = new int[4][2];
		setting(TetrisShapes.NONE);
	}
	public void setting(TetrisShapes tetris){ // each shape's coordination is defined 
		 int [][] NONE ={{0,0},{0,0},{0,0},{0,0}},
				  Z={{0,0},{0,-1},{-1,0},{-1,1}},
				 S={{0,0},{0,-1},{1,1},{1,0}},
				 I={{0,0},{0,-1},{0,1},{0,2}},
				 T= {{0,0},{-1, 0},{1,0},{0,1}},
				 SQ={{0, 0},{1,0},{0,1},{1,1}},
				 L={{0,0},{-1,-1},{0,-1},{0,1}},
				 ML={{0,0},{1,-1},{0,1},{0,-1}};
		 
		Graph = new int[][][]{NONE,Z,S,I,T,SQ,L,ML}; // the graph will include all the shapes in order
		for(int i = 0;i <4; i++){
			for(int j = 0; j<2; ++j){
				cordination[i][j] = Graph[tetris.ordinal()][i][j];
			}
		}
		TetrisPiece = tetris;
	}
	
	public int gety(int x){
		return cordination[x][1];
	}
	
	public TetrisShapes getShape(){ //return the current piece 
		return TetrisPiece;
	}
	public int getx(int x){
		return cordination[x][0];
	}
	public void sety(int pos,int y){ //  sets the shape coordination of the shape 
		cordination[pos][1] = y;
	}
	public void setx(int pos,int x){ // sets the shape coordiantion of the shape
		cordination[pos][0] = x;
	}
	 public void randomShape() // randomly picks a shape from the enum TetrisShape
	    {
	        Random rand = new Random();
	        int randomPicked = rand.nextInt(7)+1;
	        TetrisShapes[] result = TetrisShapes.values(); 
	        setting(result[randomPicked]);
	    }

	    public Shapes rotateRight() // rotate function that will mirrors the coordination of each point by mirroring the x coordination with the current y coordination
	    {
	        if (TetrisPiece == TetrisShapes.SQ)return this;
	        	
	        Shapes result = new Shapes();
	        result.TetrisPiece = TetrisPiece;

	        for (int i = 0; i < 4; ++i) {
	            result.setx(i, -(gety(i)));
	            result.sety(i, getx(i));
	        }
	        return result;
	    }
	   
	    public int miniY() // gets the lowest y coordination of the shape and returns it which in maths sense would be the x coordinations of a shape on a graph 
	    {
	      int mini = cordination[0][1];
	      for (int i=0; i < 4; i++) {
	          mini = Math.min(mini, cordination[i][1]);
	      }
	      return mini;
	    }
	
}
