package BalanceBikeGame;

import java.awt.*;
import java.awt.geom.Path2D;

public class BikeLimiters extends Bike{

    public BikeLimiters(float sx, float sy, float vx, float vy, float radius, Color col, float mass, float rollingFriction, float width, float hegiht){
        super(sx,sy,vx,vy,radius,col,mass,rollingFriction,null,4);
        this.h = hegiht;
        this.w = width;
        this.setPolygonPath(damber_Frontmaker(width,hegiht),4);

    }

    private Path2D.Float damber_chainmaker(float width, float height){

        Path2D.Float p = new Path2D.Float();
        p.moveTo(0.5*width, -0.5*height);
        p.lineTo(0.5*width, 0.5*height);
        p.lineTo(-0.5 *width, 0.5*height);
        p.lineTo(-0.5 *width, -0.5*height);
        p.lineTo(0.5 *width, -0.5*height);


        p.closePath();
        return p;

    }
    private Path2D.Float damber_Frontmaker(float width, float height){

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
    public void notificationOfNewTimestep() {

    }
}
