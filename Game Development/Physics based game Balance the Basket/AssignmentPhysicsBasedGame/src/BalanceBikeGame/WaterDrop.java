package BalanceBikeGame;

import org.jbox2d.common.Vec2;
import org.jbox2d.particle.ParticleSystem;

import java.awt.*;

import static org.jbox2d.particle.ParticleType.b2_waterParticle;

public class WaterDrop extends Shapes {

    public boolean dropped;
    public int ID;
    public WaterDrop(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, int ID) {
        super(position, velocity, radius, color, mass, rollingFriction, "waterDrop,"+ID);

        this.ID = ID;
        dropped = false;
    }


    @Override
    public void update() {
        if (rollingFriction>0) {
            Vec2 rollingFrictionForce=new Vec2(body.getLinearVelocity());
            rollingFrictionForce=rollingFrictionForce.mul(-rollingFriction*mass);
            body.applyForceToCenter(rollingFrictionForce);

        }
    }
}
