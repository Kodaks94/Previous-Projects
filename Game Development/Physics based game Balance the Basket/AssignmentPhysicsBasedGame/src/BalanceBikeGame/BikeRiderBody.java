package BalanceBikeGame;

import org.jbox2d.collision.shapes.CircleShape;
import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.BodyDef;
import org.jbox2d.dynamics.BodyType;
import org.jbox2d.dynamics.FixtureDef;
import org.jbox2d.dynamics.World;

import java.awt.*;
import java.awt.geom.Path2D;


public class BikeRiderBody extends Rider{

    boolean arm =false;
    public BikeRiderBody(Vec2 position, Vec2 velocity, float radius, Color col, float mass, float rollingFriction, float width, float hegiht){
        super(position, velocity,radius,col,mass,rollingFriction,null,4, "Limbs");
        this.h = hegiht;
        this.w = width;
        this.setPolygonPath(bodymaker(width,hegiht),4);

    }
    private Path2D.Float bodymaker(float width, float height){

        Path2D.Float p = new Path2D.Float();
        p.moveTo(-0.5*width, -0.5*height);

        for (int n = 0; n < 2; n++) {
            double x =  (-0.500 *  width);
            if (n > 0) {
                double y =  Math.abs(-0.500 *  height);
                p.lineTo(x, y);
            }
            else {
                double y = (-0.500 * height);
                p.lineTo(x, y);
            }
        }

        for (int n = 0; n < 2; n++) {
            double x =  (0.500 *  width);
            if (n > 0) {
                double y = -0.500 * height;
                p.lineTo(x, y);
            }
            else {
                double y = Math.abs(0.500 * height);
                p.lineTo(x, y);
            }
        }

        p.closePath();
        return p;

    }


    @Override
    public void update() {



    }
}
