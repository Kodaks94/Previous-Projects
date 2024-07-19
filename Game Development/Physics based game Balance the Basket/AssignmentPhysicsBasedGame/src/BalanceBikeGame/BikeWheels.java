package BalanceBikeGame;

import org.jbox2d.common.Vec2;

import java.awt.*;
import java.awt.geom.Path2D;

public class BikeWheels extends Bike{

    public int ID;

    public boolean TouchingTheGround;
    public BikeWheels(float sx, float sy, float vx, float vy, float radius, Color col, float mass, float rollingFriction, int ID) {
        super(sx, sy, vx, vy, radius, col, mass, rollingFriction);
        this.ID = ID;
        if(ID==1) this.body.setUserData("FrontBikeWheel");
        if(ID==2) this.body.setUserData("BackBikeWheel");
        TouchingTheGround = false;
    }

    @Override
    public void notificationOfNewTimestep() {
        if (rollingFriction>0) {
            Vec2 rollingFrictionForce=new Vec2(body.getLinearVelocity());
            rollingFrictionForce=rollingFrictionForce.mul(-rollingFriction*mass);
            body.applyForceToCenter(rollingFrictionForce);

        }


        if (BasicKeyListener.isRotateLeftKeyPressed() && ID == 1) {

            if(body.getLinearVelocity().x <= 500) {
                body.applyTorque(MAGNITUDE_OF_ENGINE_THRUST_FORCE);
            }
        }
       else if (BasicKeyListener.isRotateRightKeyPressed()&& ID == 2) {
            if(body.getLinearVelocity().x <= 500) {
                body.applyTorque(-MAGNITUDE_OF_ENGINE_THRUST_FORCE);
            }
        }

    }
}
