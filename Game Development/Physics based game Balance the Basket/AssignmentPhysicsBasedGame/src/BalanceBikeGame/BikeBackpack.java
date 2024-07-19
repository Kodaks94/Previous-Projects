package BalanceBikeGame;

import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.joints.WeldJoint;
import org.jbox2d.dynamics.joints.WeldJointDef;

import java.awt.*;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BikeBackpack {

    List<WaterDrop> waterList;
    Map<String,backpackComponent> backpackList;
    public BikeBackpack(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, Path2D.Float polgonPath, int numSides, float height, float width) {
       backpackList = new HashMap<>();
       backpackComponent left = new backpackComponent(position,velocity,radius,color,mass,rollingFriction,polgonPath,6, height,width,1,"backpack");
       backpackList.put("Left",left);
        backpackComponent right = new backpackComponent(position,velocity,radius,color,mass,rollingFriction,polgonPath,6, height,width,2,"backpack");
       backpackList.put("Right",right);
       backpackComponent bottom = new backpackComponent(position,velocity,radius,color,mass,rollingFriction,polgonPath,6, height,width,3,"backpack");
        backpackList.put("Bottom",bottom);
        backpackComponent top = new backpackComponent(position,velocity,radius,color,mass,rollingFriction,polgonPath,6, height,width,4,"backpack");
        backpackList.put("Top",top);

        waterList =  new ArrayList<>();

        for(int i = 0; i < 50; i ++){

            WaterDrop b  = new WaterDrop( new Vec2(( right.body.getPosition().x - left.body.getPosition().x)/2 + left.body.getPosition().x - 1f, top.body.getPosition().y + 1.5f),new Vec2( 0,0),0.09f,Color.BLUE,0.00000000001f, 2f, i);

            waterList.add(b);

        }

        WeldJointDef weldJointDef = new WeldJointDef();

        weldJointDef.bodyA = left.body;
        weldJointDef.bodyB = bottom.body;
        weldJointDef.localAnchorA.set(new Vec2(left.body.getLocalCenter().x, left.body.getLocalCenter().y - left.h/2));
        weldJointDef.localAnchorB.set(new Vec2(bottom.body.getLocalCenter().x - bottom.w/2, bottom.body.getLocalCenter().y ));
        CONSTANTS.world.createJoint(weldJointDef);
        weldJointDef.bodyA = right.body;
        weldJointDef.localAnchorB.set(new Vec2(bottom.body.getLocalCenter().x - bottom.w/2 - 0.2f, bottom.body.getLocalCenter().y ));
        CONSTANTS.world.createJoint(weldJointDef);
        weldJointDef.bodyB = left.body;
        weldJointDef.localAnchorA.set(new Vec2(bottom.body.getLocalCenter().x + 0.1f , right.body.getLocalCenter().y - right.h/2));
        weldJointDef.localAnchorB.set(new Vec2(bottom.body.getLocalCenter().x - 0.1f, left.body.getLocalCenter().y - left.h/2));
        CONSTANTS.world.createJoint(weldJointDef);
        weldJointDef.bodyB = top.body;

        weldJointDef.localAnchorB.set(new Vec2(top.body.getLocalCenter().x +top.w/2, top.body.getLocalCenter().y));
        weldJointDef.localAnchorA.set(new Vec2(top.body.getLocalCenter().x +top.w/2, top.body.getLocalCenter().y));
        CONSTANTS.world.createJoint(weldJointDef);



    }
    public void addtoDropped(String userData){

        for(WaterDrop d : waterList){

            if(d.body.getUserData().toString().equals(userData) && !d.dropped){

                d.dropped = true;

            }

        }


    }
    public int dropCount(){
        int i = 0;
        for(WaterDrop d : waterList) {
        if(d.dropped){
            i++;
        }
        }
        return i;

    }


    public class backpackComponent extends Shapes{

        //LEFT 0, RIGHT 1
        private int side;
        public backpackComponent(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, Path2D.Float polgonPath, int numSides, float height, float width, int side, String userData) {
            super(position, velocity, radius, color, mass, rollingFriction, polgonPath, numSides,userData);
            this.h = height;
            this.w = width;
            this.side = side;
            this.setPolygonPath(backpackmaker(width,height),4);
        }


        public Path2D.Float backpackmaker(float width, float height){
            Path2D.Float p = new Path2D.Float();
            if(side == 1) {
                p.moveTo(-0.5 * width, -0.5 * height);
                p.lineTo(-0.5 * width, 0.5 * height);
                p.lineTo(-0.7 * width, 0.5 * height);
                p.lineTo(-0.7 * width, -0.5 * height);
                p.lineTo(-0.5 * width, -0.5 * height);
                //p.lineTo(0 * width, -0.5 * height);
                //p.lineTo(-0.5 * width, -0.5 * height);
            }
            else if(side == 2){

                p.moveTo(0.5 * width, -0.5 * height);
                p.lineTo(0.5 * width, 0.5 * height);
                p.lineTo(0.7 * width, 0.5 * height);
                p.lineTo(0.7 * width, -0.5 * height);
                p.lineTo(0.5 * width, -0.5 * height);
               // p.lineTo(0 * width, -0.5 * height);
                //p.lineTo(0.5 * width, -0.5 * height);
            }
            else if(side == 3){
                p.moveTo(-0.5 * width, -0.5 * height);
                p.lineTo(-0.5 * width, -0.7 * height);
                p.lineTo(+0.5 * width, -0.7 * height);
                p.lineTo(0.5 * width, -0.5 * height);
                p.lineTo(-0.5 * width, -0.5 * height);

            }
            else if(side==4){
                p.moveTo(+0.5 * width, 0.5 * height);
                p.lineTo(-0.2* width , 0.5*height);
                p.lineTo(-0.2 * width, 0.7 * height);
                p.lineTo(+0.5 * width, 0.7 * height);
                p.lineTo(0.5*width, 0.5* height);
            }

            p.closePath();
            return p;


        }


        @Override
        public void update() {

            for(WaterDrop w : waterList){

                w.update();
            }

        }
    }
}
