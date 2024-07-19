package BalanceBikeGame;

import org.jbox2d.common.Vec2;

import java.awt.*;

public class BikeRiderHead extends Rider {
    public BikeRiderHead(Vec2 position, Vec2 velocity, float radius, Color col, float mass, float rollingFriction) {
        super(position, velocity, radius, col, mass, rollingFriction,"Head");
    }



    @Override
    public void update() {

    }
}
