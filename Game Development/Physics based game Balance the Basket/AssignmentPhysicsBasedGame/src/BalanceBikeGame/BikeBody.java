package BalanceBikeGame;

import org.jbox2d.common.Vec2;

import java.awt.*;
import java.awt.geom.Path2D;

public class BikeBody extends Bike {


    public BikeBody(float sx, float sy, float vx, float vy, float radius, Color col, float mass, float rollingFriction, float width, float hegiht, int numSides) {

        super(sx, sy, vx, vy, radius, col, mass, rollingFriction, null, numSides);
        this.setPolygonPath(create_BikeBody(width, hegiht), 4);
        this.h= hegiht;
        this.w = width;

    }


    private Path2D.Float create_BikeBody(float width, float height) {


        Path2D.Float p = new Path2D.Float();
        p.moveTo(0.5 * width + 0.8, 0.5 * height);
        //B to A
        p.lineTo(-0.5 * width, 0.3 * height);
        //A to C
        p.lineTo(-0.5 * width, -0.5 * width);
        //C to D
        p.lineTo(0.5 * width, -0.5 * height);
        //D to B
        p.lineTo(0.5 * width + 0.8, 0.5 * height);


        p.closePath();
        return p;

    }


    @Override
    public void notificationOfNewTimestep() {
       if (rollingFriction > 0) {
            Vec2 rollingFrictionForce = new Vec2(body.getLinearVelocity());
            rollingFrictionForce = rollingFrictionForce.mul(-rollingFriction * mass);

            body.applyForceToCenter(rollingFrictionForce);

        }

        if (BasicKeyListener.isRotateLeftKeyPressed()) {
            if(body.getLinearVelocity().x <= 800) {
                body.applyTorque(-MAGNITUDE_OF_BODY_THRUST_FORCE);
            }

    }
        if (BasicKeyListener.isRotateRightKeyPressed()) {
            if(body.getLinearVelocity().x <= 800) {

                body.applyTorque(MAGNITUDE_OF_BODY_THRUST_FORCE * mass * 8);
            }
        }

    }
}
