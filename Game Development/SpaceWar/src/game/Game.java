package game;

import static game.Constants.DELAY;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.JOptionPane;
import utilities.JEasyFrame;

public class Game {
	  public static final int N_INITIAL_ASTEROIDS = 5;
	  public List<GameObject> objects;
	 public  Ship ship;
	  public KeyControl ctrl;
	  public static int level = 1;
	  public Ammo ammo;
	  private Bullet bull;
	  private HashMap<String,Boolean> levels = new HashMap<String,Boolean>();
	  public Score sc;
	  int [] maxScorelvl={15,21,21,30,58};
	  List<Integer> usedScore = new ArrayList<Integer>();
	  public Asteroid as;
	  public Cannon cannon;
	  int count = 0;
	  public Game() {
		  setLevel();
		  objects = new ArrayList<GameObject>();
		  
		for (int i = 0; i < N_INITIAL_ASTEROIDS; i++) {
		     Asteroid.makeRandomAsteroid(); 
		  }
		 for(int i = 0; i < 6; i++){
			Cannon.RandomlyGeneratedCannons();
		 }
		  
		 ammo = new Ammo();
		 	sc = new Score();
		  ctrl= new KeyControl(); 
		  ship = new Ship(ctrl);
		 
		  objects.add(ship);
		  objects.add(sc);
		  objects.add(ammo);
		  
		}

	  public static void main(String[] args) throws Exception {
		  Game game = new Game();
		  View view = new View(game);
		  new JEasyFrame(view, "Basic Game").addKeyListener(game.ctrl);
		  // run the game
		  while (true) {
		    game.update();
		    view.repaint();
		    Thread.sleep(DELAY);
		  }
		}
	  
	  public void update() {
		 sc.updatelevel(level);
		 Levels();
		 levelRules();
		 SetMutation();
		 addingCannons(objects);
		 addingAsteroids(objects);
		 addingBullets(objects);
		 addingPowers(objects);
		  
		if(Ship.lives == 0){
			GameOver();
		}
		
		 
		
		List<GameObject> alive = new ArrayList<>();
		List<GameObject> tempalive= new ArrayList<>();
		for(GameObject ob : objects){
			ob.update();
			if(!ob.dead)tempalive.add(ob);
		}
		
		CollisionRules(tempalive);
		
		for(GameObject ob: tempalive){
			if(!ob.dead)alive.add(ob);
		}
		incScore(tempalive);
		synchronized(Game.class){
			objects.clear();
			objects.addAll(alive);
			
		}
		
	}
	  
	  private void levelRules(){
		
		 for(int i : maxScorelvl){
			 if( getScore()== i && !usedScore.contains(i)){
				 usedScore.add(i);
				 Score.sc = 0;
				
				 lvlup();
				 
				 
			 }
		 }
		 
	  }
	  private void lvlup(){
		  level++;
		  
		  
		  resetComponents();
		  if(level > 5){
			  GameOver();
		  }
	  }
	  private void GameOver(){
		  objects.clear();
		  ScoreBoard();
		  System.exit(1);
	  }
	  private void ScoreBoard(){
		  String[] buttons = {"Save","Cancel"};
          String Player = null;
          int rc = JOptionPane.showOptionDialog(null, "Would you like to save?", "Save", JOptionPane.WARNING_MESSAGE, 0, null, buttons, buttons[1]);
          if(rc == 0){
          	Player = JOptionPane.showInputDialog("enter Name");
          	Save s = new Save();
          	s.saving(Player,String.valueOf(Score.totalsc));
          	s.Loading();
          }
	  }
	  private void ShipDeath(){
		  if(Ship.lives >0){
				 ship.reset();
				 Ship.lives--;
				 ship.dead = false;
			 }
	  }
	  
	  private void CollisionRules(List<GameObject> obs){
		  for(GameObject ob : obs){
			  for(GameObject other : obs){
				  if(ob instanceof Ship && other instanceof Asteroid) {
					  if(ob.CollisionHandling(other)){
						  ShipDeath();
						  Score.sc += 2;
						  Score.totalsc += 2;
					  }
				  }
				  if(ob instanceof Ship&& other instanceof Cannon){
					  if(ob.CollisionHandling(other)){
						  ShipDeath();
					  }
				  }
				  if(ob instanceof Bullet && other instanceof Ship){
					  if(ob.BulletReferernce == 2){
						 if( ob.CollisionHandling(other)){
							 ShipDeath();
						 }
					  }
				  }
				  if(ob instanceof Bullet && other instanceof Asteroid){
					  if(ob.BulletReferernce == 1){
						  if(ob.CollisionHandling(other)){
							  mutation(other);
							  PowerUps.powerdrop(other);
					  }
						  
					  }	  
				  }
				  
				  if(ob instanceof Bullet && other instanceof Cannon  ){
					  if(ob.BulletReferernce ==1) {
						  if(ob.CollisionHandling(other)){
							  PowerUps.powerdrop(other);
						  }
					  }
					  }
				  if(ob instanceof Ship && other instanceof PowerUps){
					  if(ob.CollisionBoolean(other)){
						  other.dead = true;
						 if(other.power == 1){
							 Ship.lives++;
						 }
						 else if(other.power == 2){
							 GameObject.powered = true;
							 
						 }
					  }
				  }
				  }
			  }
		  }
	  private void setLevel(){
		  levels.put("Asteroid", false);
		  levels.put("Canon", false);
		  levels.put("Asteroidfive", false);
		  levels.put("Asteroidten", false);
	  }
	  private void Levels(){
		  if(level ==1){
			  for(String a: levels.keySet()){
				  if(a.equals("Asteroid")) levels.put("Asteroid", true);
				  else levels.put(a, false);
			  }
		  }
		  else if(level ==2){
			  
			  for(String a: levels.keySet()){
				  if(a.equals("Asteroid")) levels.put(a, true);
				  else if(a.equals("Canon")) levels.put(a , true);
				  else levels.put(a, false);
			  }
		  }
		  else if(level ==3){
			  for(String a: levels.keySet()){
				  if(a.equals("Asteroid")) levels.put(a, true);
				  else if(a.equals("MovingCanon")) levels.put(a , true);
				  else levels.put(a, false);
			  }
		  }
		  else if(level ==4){
			  for(String a: levels.keySet()){
				  if(a.equals("Asteroidfive")) levels.put(a, true);
				  else if(a.equals("Cannon")) levels.put(a , true);
				  else levels.put(a, false);
			  }
		  }
		  else if(level ==5){
			  for(String a: levels.keySet()){
				  if(a.equals("Asteroidten")) levels.put(a, true);
				  else if(a.equals("Cannon")) levels.put(a , true);
				  else levels.put(a, false);
			  }
		  }
	  }
	  
	  private void SetMutation(){
		  if(levels.get("Asteroid")){
				Asteroid.Multiply =2;
				
			}
		  else if(levels.get("Asteroidfive")){
			  Asteroid.Multiply = 5;
			}
		  else if(levels.get("Asteroidten")){
				Asteroid.Multiply = 10;
			}
	  }
	  public  void mutation(GameObject a ){
		 Asteroid.mutation(a);
		  
}
	
	  public void incScore(List<GameObject> obs){
		  for(GameObject ob: obs ){
			  if((ob instanceof Asteroid || ob instanceof Cannon) && ob.dead){
				  Score.sc++;
				  Score.totalsc++;
			  }
		  }
	  }
	  public int getScore(){
		  return Score.sc;
	  }
	  private void addingPowers(List<GameObject>obs){
		  if(!PowerUps.powers.isEmpty()){
			  for(PowerUps a: PowerUps.powers){
				  if(!obs.contains(a)){
					  if(!a.dead)
					  obs.add(a);
				  }
			  }
		  }
	  }
	  
	  private void addingBullets(List<GameObject> obs){
		  if(!Bullet.bl.isEmpty()){
			  for(Bullet b : Bullet.bl){
				 
				  obs.add(b);
			  
			  }
		  }
		  if(!Bullet.cbl.isEmpty()){
			  for(Bullet b : Bullet.cbl){
				 
				  obs.add(b);
			  
			  }
		  }
	  }
	  private void addingAsteroids(List<GameObject> obs){
		
		 
			 
		  if(!Asteroid.SpawnedAsteroids.isEmpty()){
			  
				for(Asteroid a: Asteroid.SpawnedAsteroids){
						
						if(!obs.contains(a)){
							
							if(!a.dead)
							obs.add(a);
						
						}
						
							
				}
		  }
		  }
		  
		
	  
	  private void resetComponents(){
		 Asteroid.usedAsteroid.clear();
		 Asteroid.SpawnedAsteroids.clear();
		 for (int i = 0; i < N_INITIAL_ASTEROIDS; i++) {
		     Asteroid.makeRandomAsteroid(); 
		  }
		  ship.reset();
	  }
	  private void addingCannons(List<GameObject> obs){
		 if(levels.get("Canon")){
		  if(!Cannon.cannons.isEmpty()){
			  for(Cannon a: Cannon.cannons){
				  if(!obs.contains(a)){
						if(!a.dead)
						obs.add(a);
						
					}
				  a.updateShip(ship);
			  }
		  }
	  }
	  }
}