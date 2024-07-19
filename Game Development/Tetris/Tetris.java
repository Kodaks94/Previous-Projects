 package myTetris;


import java.awt.BorderLayout;
import java.util.Arrays;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JLabel;


public class Tetris
{
    public static void main(String[] args) // the main functuion where the program will execute 
    {										// which calls the framing class to set the window visibility to true
    	
    	Framing frame = new Framing();   
    	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    	frame.setVisible(true);
      
    }

	
}