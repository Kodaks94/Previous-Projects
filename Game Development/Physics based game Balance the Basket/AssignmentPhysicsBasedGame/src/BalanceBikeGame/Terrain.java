package BalanceBikeGame;

import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.BodyType;

import java.awt.*;
import java.awt.geom.Path2D;

public abstract class Terrain extends Shapes {

    public Terrain(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, Path2D.Float polgonPath, int numSides) {
        super(position, velocity, radius, color, mass, rollingFriction, polgonPath, numSides, "Ground");
        this.body.setType(BodyType.KINEMATIC);
    }



    public enum TerrainType{
        Hard (6,22,5,4.5f),
        Easy (6,20,5,4.5f),
        Ramp(3, 5, 3,3.5f),
        Complex (6,20,5,4.5f),
        Steep (5,20,5,4.5f);

        private final int numsides;
        private final float width, height;
        private final float y_position;
        private TerrainType(int numsides, float width, float height, float y_position){
            this.numsides = numsides;
            this.width = width;
            this.height = height;
            this.y_position = y_position;
        }

        public float getY_position() {
            return y_position;
        }

        public float getWidth(){
            return width;
        }

        public float ActualWidth(){

            if(this == Complex){
                float newwidth = width*2;
                return newwidth;

            }
            return width;

        }

        public float getHeight() {
            return height;
        }

        public int getNumsides() {
            return numsides;
        }
        public static TerrainType[] getEasyTerrains(){

            return new TerrainType[]{Easy, Ramp};
        }
        public static TerrainType[] getHardTerrains(){
            return new TerrainType[]{ Steep, Hard};
        }
    }
    public static class TerrainShape extends  Terrain{

        private  TerrainType type;

        public TerrainShape(float x, Color color, TerrainType type) {
            super(new Vec2(x, type.getY_position()), new Vec2(0,0), type.getWidth(), color, 0f, 2f, null, type.getNumsides());
            this.type = type;

            this.setPolygonPath(TerrainMaker(type.getWidth(),type.getHeight()), type.getNumsides());

        }



        private Path2D.Float TerrainMaker(float width, float height){

            Path2D.Float p = new Path2D.Float();

            p.moveTo(-0.5* width, -0.5* height);

            switch (type){

                case Hard:
                    p.lineTo(-0.27*width,0);
                    p.lineTo(-0.15 * width, 0.5 * height);
                    p.lineTo(0.15 *width, 0.5* height);
                    p.lineTo(0.27*width,0);
                    p.lineTo(0.5*width, -0.5*height);
                    p.lineTo(-0.5*width, -0.5* height);

                    break;
                case Easy:
                    p.lineTo(-0.27* width, 0*height);
                    p.lineTo(-0.10 *width, 0*height);
                    p.lineTo(-0.01 * width, 0.3*height);
                    p.lineTo(0.15 * width, 0.3*height);
                    p.lineTo(0.5*width, -0.5*height);
                    p.lineTo(-0.5*width, -0.5*height);
                    break;
                case Ramp:
                    p.lineTo(-0.5 * width, -0.5 * height);
                    p.lineTo(0.5 * width, 0.5 * height);
                    p.lineTo(0.5 * width, -0.5 * height);
                    break;
                case Complex:

                    p.moveTo(-0.5 * width, -0.5* height);
                    p.lineTo(-0.3* width, 0 * height);
                    p.lineTo(-0.2 * width, 0.5*height);
                    p.lineTo(-0.1 * width, 0.2 * height);
                    p.lineTo(-0.01 * width, 0.5* height);
                    p.lineTo(0*width, 0.4* height);
                    p.lineTo(0.1*width, 0.5*height);
                    p.lineTo(0.2*width, 0);
                    p.lineTo(0.3*width, 0.5*height);
                    p.lineTo(0.5*width, -0.5*height);
                    p.lineTo(-0.5*width, -0.5* height);

                    break;
                case Steep:
                    p.lineTo(-0.27*width, 0.2*height);
                    p.lineTo(-0.10*width, 0.5*height);
                    p.lineTo(0.15* width, 0.5*height);
                    p.lineTo(0.5*width, -0.5*height);
                    p.lineTo(-0.5*width, -0.5*height);


                    break;
            }

            p.closePath();

            return p;

        }




        @Override
        public void update() {

        }
    }

}
