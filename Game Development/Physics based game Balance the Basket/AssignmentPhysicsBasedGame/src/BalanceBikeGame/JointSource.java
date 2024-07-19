package BalanceBikeGame;

import java.awt.*;

public class JointSource extends Bike
{


    public JointSource(float sx, float sy, float vx, float vy, float radius, Color col, float mass, float rollingFriction) {
        super(sx, sy, vx, vy, radius, col, mass, rollingFriction);
    }

    @Override
    public void notificationOfNewTimestep() {

    }
}
