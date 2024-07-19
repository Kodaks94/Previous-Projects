package BalanceBikeGame;

import javafx.scene.transform.Scale;
import org.jbox2d.collision.shapes.CircleShape;
import org.jbox2d.collision.shapes.PolygonShape;
import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.*;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Path2D;
import java.awt.geom.PathIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Clouds {


    List<cloudData> cloudDataList;

    Vec2 StartingPos;
    public Clouds(Vec2 StartingPos){
        cloudDataList = new ArrayList<>();
        this.StartingPos = StartingPos;
        setupTheClouds(50);

    }


    public void setupTheClouds(int numClouds){

        for(int i =0; i < numClouds; i++){
            Random _random = new Random();
            Vec2 jitter = new Vec2((float)(_random.nextDouble() * 100 - 50), (float)(_random.nextDouble()) - 0.5f);

            Vec2 newpos = StartingPos.add(new Vec2(StartingPos.x+ i*40 , 0));
            cloudData c = new cloudData(newpos,0,Color.CYAN, i);
            c.radius = c.MAX_radius/(i+1);
            cloudDataList.add(c);
        }

    }
    public void updateClouds(){


        for (cloudData c : cloudDataList){

           c.ScalingCheck();
           c.update();


        }



    }
    public void draw(Graphics2D g){

        for (cloudData c : cloudDataList) {
            c.draw(g);
        }

    }



enum Scale{
        ScaleDown,
        ScaleUp,

}
    class cloudData{

        Vec2 position;
        float radius;
        Color color;
        int ID;
        int ScaleFactor = 0;
        int MAX_radius = 40;
        private Scale scaleStatus = Scale.ScaleUp;
        public Scale ScalingCheck(){
            if(radius <= 1){
                position = position.add(new Vec2(MAX_radius, 0));
                scaleStatus = Scale.ScaleUp;
                return Scale.ScaleUp;
            }
            if(radius >= MAX_radius){
                scaleStatus = Scale.ScaleDown;
                return Scale.ScaleDown;
            }
            else return scaleStatus;
        }
        public cloudData(Vec2 position, float radius, Color color, int ID ){

            this.position = position;
            this.radius = radius;
            this.color = color;
            this.ID = ID;


        }
        public void update() {

            switch (scaleStatus) {

                case ScaleDown:
                    radius -= 0.5;
                    break;
                case ScaleUp:
                    radius += 0.5;
                    break;

            }
        }


        public void draw(Graphics2D g){
            g.setColor(color);

            for(int i = 0 ; i < 5; i++){
                Random _random = new Random();
                Vec2 jitter = new Vec2((float)(_random.nextDouble() * 5 - 1), (float)(_random.nextDouble()) - 0.5f);
                Vec2 newPos = new Vec2(position.x + jitter.x, position.y + jitter.y);
                g.fillOval((int)newPos.x, (int)newPos.y,(int)radius,(int)radius);

            }





        }




    }



}
