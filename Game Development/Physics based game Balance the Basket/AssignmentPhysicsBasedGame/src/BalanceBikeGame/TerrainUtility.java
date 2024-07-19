package BalanceBikeGame;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class TerrainUtility {

    private List<Terrain> Terrains;

    public TerrainUtility(){

        Terrains = new ArrayList<>();
    }

    public void add_to_TerrainList(Terrain terrain){

        Terrains.add(terrain);

    }



    public void draw(Graphics2D g){

        for(Terrain t: Terrains){
            t.draw(g);

        }
    }


    public void update(){

        for (Terrain t: Terrains){
            t.update();
        }


    }


}
