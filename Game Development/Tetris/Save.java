package myTetris;

import java.io.*;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;

import javax.swing.JOptionPane;

public class Save { // this class will save the scores if the user wishes to save their name 
	private File Score ;

	
	String [][] Players;
	Save(){
	
 }
	public void saving(String newPlayer, String Scores){ // this  function will get the name of the player and score and opens or creates the new text file and write and saves their name 
		// line by line 
		try{
			Score = new File("Scores.txt");
		BufferedWriter output;
		String input = newPlayer + Scores+"\n";
		output = new BufferedWriter(new FileWriter(Score,true));
		output.append(input);
		output.newLine();
		output.close();
		}
		catch(Exception e){
			System.out.println(e.getMessage()+"save");
		}
}
	public void Loading(){ // loads the score from the text file and reads them line by line and shows it on the JOptionPane
		try{
		Score = new File("Scores.txt");
		FileReader reader = new FileReader(Score);
		BufferedReader bf = new BufferedReader(reader);
		StringBuffer sb = new StringBuffer();
		String line;
		String [] temp = null;
		String Scores = "Scores"+"\n";
		while((line = bf.readLine()) != null){
			
			Scores += line+"\n";
			
		}
		reader.close();
		JOptionPane.showMessageDialog(null,Scores );
		}
		catch(Exception e){
			System.out.print(e.getMessage()+"load");
		}
	}
}
