package game;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import javax.swing.JOptionPane;

public class Save { 
	private File Score ;

	
	String [][] Players;
public Save(){
	
 }
	public void saving(String newPlayer, String Scores){ 
		
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
	public void Loading(){ 
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
