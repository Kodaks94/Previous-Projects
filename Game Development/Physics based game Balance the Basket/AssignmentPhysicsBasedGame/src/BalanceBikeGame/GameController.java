package BalanceBikeGame;

import java.util.*;

public class GameController {

    public int level;
    private int num_bonus_points;
    public int Score;
    private int applybonus;

    Map<Integer, List<Terrain.TerrainType>> levels;
    public GameController(int Num_levls){

        level = 1;
        num_bonus_points = 50;
        Score = 0;
        levels = new HashMap<>();
        setuplevels(Num_levls);
        applybonus = 1;

    }


    public void reset(){
        applybonus = 1;


    }
    public void setuplevels(int num_of_Levels){

        for(int i = 1; i <= num_of_Levels; i++){
            List<Terrain.TerrainType> terrains = new ArrayList<>();
            Random random = new Random();
            for (int j =1; j <= i; j++){

                if(j <3){


                    terrains.add(Terrain.TerrainType.getEasyTerrains()[random.nextInt(Terrain.TerrainType.getEasyTerrains().length)]);
                }
                else{
                    terrains.add(Terrain.TerrainType.getHardTerrains()[random.nextInt(Terrain.TerrainType.getHardTerrains().length)]);
                }
            }


            levels.put(i, terrains);
        }
    }
    public void removeBonus(){
        applybonus = 0;
    }
   public void calculateScore( boolean won, int pointslost){
        if(won) {

            Score += ((level * 10) + applybonus*((num_bonus_points - pointslost)));

        }
        else{
            Score = 0;
        }

   }

   public float LevelWidth(){

        float width = 0;
        for(Terrain.TerrainType t: levels.get(level)){

            width+= t.getWidth();
        }
        return width;

   }
   public void newGame(){
        level++;
   }






}
