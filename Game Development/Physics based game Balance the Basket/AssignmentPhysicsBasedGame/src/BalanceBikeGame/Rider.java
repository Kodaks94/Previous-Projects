package BalanceBikeGame;

import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.joints.*;

import java.awt.*;
import java.awt.geom.Path2D;
import java.util.*;
import java.util.List;

public abstract class Rider extends Shapes  {

    List<Rider> riderList = new ArrayList<Rider>();

    public Rider(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, Path2D.Float polgonPath, int numSides,String userData) {
        super(position, velocity, radius, color, mass, rollingFriction, polgonPath, numSides,userData);
    }

    public Rider(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, String userData) {
        super(position, velocity, radius, color, mass, rollingFriction,userData);
    }


    public static Map<String, Rider> Create_the_Rider(Vec2 Centalpos, float headR, float width, float height){

        Vec2 positionhead = new Vec2(Centalpos.x+ width/2, Centalpos.y + width/2);
        Vec2 velocity = new Vec2();
        BikeRiderHead  head =  new BikeRiderHead(positionhead, velocity,headR,Color.green, 0.01f,9f);
        BikeRiderBody body = new BikeRiderBody(Centalpos, velocity,headR*4, Color.cyan, 0.1f, 9f, width/2, height*2);
        Vec2 armpos = new Vec2(body.body.getWorldCenter().x, body.body.getWorldCenter().y - body.h /3);
        BikeRiderBody arms = new BikeRiderBody(armpos, velocity, headR *2 , Color.WHITE, 0.01f, 9f, width / 2, height*1.5f);
        armpos = armpos.add(new Vec2(0.1f,0));
        BikeRiderBody arms2 = new BikeRiderBody(armpos, velocity, headR *2 , Color.WHITE, 0.01f, 9f, width / 2, height*1.5f);
        arms.arm = true;
        //
        Vec2 Legpos = armpos.add(new Vec2(0,body.h/2));

        BikeRiderBody leg1 = new BikeRiderBody(Legpos, velocity, headR *3, Color.WHITE, 0.01f, 9f, width/2, height*1.5f);
        BikeRiderBody leg2 = new BikeRiderBody(Legpos, velocity, headR *3, Color.WHITE, 0.01f, 9f, width/2, height/2);
        Map<String, Rider> riderparts = new HashMap<>();
        riderparts.put("Head",head);
        riderparts.put("Arm", arms);
        riderparts.put("Leg",leg1);
        riderparts.put("Body", body);



        RevoluteJointDef revoluteJoint = new RevoluteJointDef();
        revoluteJoint.bodyB = body.body;
        revoluteJoint.bodyA = arms.body;
        revoluteJoint.localAnchorA = new Vec2(arms.body.getLocalCenter().x, arms.body.getLocalCenter().y + arms.h/2);
        revoluteJoint.localAnchorB = new Vec2(body.body.getLocalCenter().x,body.body.getLocalCenter().y + body.h/2);
        revoluteJoint.collideConnected = false;
        revoluteJoint.enableLimit = true;
        revoluteJoint.upperAngle = (float) Math.toRadians(50);
        revoluteJoint.lowerAngle = (float) Math.toRadians(-90);
        CONSTANTS.world.createJoint(revoluteJoint);
        revoluteJoint.bodyA = leg1.body;
        revoluteJoint.localAnchorA = new Vec2(leg1.body.getLocalCenter().x, leg1.body.getLocalCenter().y + leg1.h/2);
        revoluteJoint.localAnchorB = new Vec2(body.body.getLocalCenter().x,body.body.getLocalCenter().y - body.h/2);
        CONSTANTS.world.createJoint(revoluteJoint);

        revoluteJoint.bodyA = head.body;
        revoluteJoint.localAnchorA.set(new Vec2(head.body.getLocalCenter()));
        revoluteJoint.localAnchorB.set(new Vec2(body.body.getLocalCenter().x, body.body.getLocalCenter().y + (body.h/2)+ 0.5f));
        revoluteJoint.upperAngle = (float) Math.toRadians(0);
        revoluteJoint.lowerAngle = (float) Math.toRadians(0);
        CONSTANTS.world.createJoint(revoluteJoint);
        return riderparts;
    }

}
