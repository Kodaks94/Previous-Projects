package BalanceBikeGame;

import java.awt.*;


import com.sun.org.apache.xpath.internal.operations.Bool;
import com.sun.xml.internal.bind.v2.runtime.reflect.opt.Const;
import javafx.util.Pair;
import org.jbox2d.callbacks.ContactImpulse;
import org.jbox2d.callbacks.ContactListener;
import org.jbox2d.collision.Manifold;
import org.jbox2d.collision.shapes.PolygonShape;
import org.jbox2d.collision.shapes.ShapeType;
import org.jbox2d.common.Color3f;
import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.World;
import org.jbox2d.dynamics.contacts.Contact;
import org.jbox2d.dynamics.joints.*;
import org.jbox2d.particle.ParticleColor;
import org.jbox2d.particle.ParticleGroupDef;
import org.jbox2d.particle.ParticleSystem;

import java.awt.*;
import java.util.*;
import java.util.List;

import static org.jbox2d.particle.ParticleGroupType.b2_solidParticleGroup;
import static org.jbox2d.particle.ParticleType.b2_elasticParticle;
import static org.jbox2d.particle.ParticleType.b2_waterParticle;



public class PhysicsEngine {

    public List<Bike> bikeparts;
    public List<AnchoredBarrier> barriers;
    public List<Shapes> shapes;
    Map<String, Rider> riderparts;
    ParticleSys sys;
    Clouds clouds;
    TerrainUtility terrainUtility;
    public List<Imp_Joint> jointDefMap;
    public GameController gm;
    BikeBackpack backpack;

    private void initialise_world(){


        jointDefMap = new ArrayList<>();
        bikeparts = new ArrayList<>();
        shapes = new ArrayList<>();
        barriers = new ArrayList<>();
        CONSTANTS.world = new World(new Vec2(0, -CONSTANTS.GRAVITY));// create Box2D container for everything

        CONSTANTS.world.setContinuousPhysics(true);
        Vec2 initialisestate = new Vec2(4, 4);
        float level_width =gm.LevelWidth() + 40 ;
        setBoundries(level_width);
        creatingBIKES(initialisestate);
        Create_bike_backpack(initialisestate);
        CreateTerrrains(initialisestate, level_width, gm);
        clouds = new Clouds(new Vec2(0, CONSTANTS.WORLD_HEIGHT/2));
        Apply_contact_rules();
        System.out.println("TOTAL"+ jointDefMap.size());


    }

    private void setBoundries(float level_width){
        barriers.add(new AnchoredBarrier_StraightLine(0,2,level_width,2,Color.WHITE));
        barriers.add(new AnchoredBarrier_StraightLine(0, CONSTANTS.WORLD_HEIGHT, 0, 0, Color.WHITE));
        barriers.add(new AnchoredBarrier_StraightLine(level_width, CONSTANTS.WORLD_HEIGHT, level_width,0,Color.WHITE));

    }
    private void creatingBIKES(Vec2 initialisestate){

        float x = initialisestate.x;
        float y=  initialisestate.y;
        float OFFSET = 0.5f;
        float OFFSETY = 0.77f;
        Bike bikebody = new BikeBody(x,y,0,0,4,Color.BLUE, .5f, 2f, 1f,1f, 4);
        Bike bike_FrontWheel = new BikeWheels(x+(OFFSET* 2.4f), y - OFFSETY, 0,0,.5f,Color.WHITE,1.5f, 2f,1);
        Bike bike_FrontDamper = new BikeLimiters(x+(OFFSET *2.4f),y-(OFFSETY/6), 0,0,.5f,Color.YELLOW,1,2f,0.5f,1.3f);
        Bike bike_BackWheel = new BikeWheels(x-(OFFSET*2.4f),y - OFFSETY, 0,0,.5f,Color.WHITE,1.5f, 2f,2);
        Bike pointerFront = new BikeLimiters(bikebody.body.getWorldCenter().x + bikebody.w, bikebody.body.getWorldCenter().y + bikebody.h/2, 0, 0, 0.5f,Color.RED,1,2f,0.3f,0.5f);
        Bike bike_BackDamper = new BikeLimiters(x-  OFFSET, y- OFFSETY+0.2f, 0,0,0.5f,Color.cyan,0.00001f,2f,1.4f,0.5f);
        Bike bike_Engine = new BikeLimiters(x ,bikebody.body.getWorldCenter().y - bikebody.h /2,0,0,1f, Color.RED, 0.00001f, 2f, 1f, 0.5f);
        bikeparts.add(bikebody);
        bikeparts.add(bike_Engine);
        bikeparts.add(bike_FrontWheel);
        bikeparts.add(bike_BackWheel);
        bikeparts.add(pointerFront);
        bikeparts.add(bike_FrontDamper);
        bikeparts.add(bike_BackDamper);
        FrontJoint(bikebody, bike_FrontWheel,bike_FrontDamper,pointerFront);
        BackJoint(bikebody, bike_BackWheel,bike_Engine, bike_BackDamper);
        ADDEDJOINTS1(bikebody,bike_BackWheel);
        ADDEDJOINTS2(bikebody,bike_FrontWheel);
        riderparts = Rider.Create_the_Rider(new Vec2(x, y+2), .5f, 1f,1f);

        RevoluteJointDef revoluteJointDef = new RevoluteJointDef();
        revoluteJointDef.bodyA =  riderparts.get("Arm").body;
        revoluteJointDef.bodyB =pointerFront.body;
        revoluteJointDef.localAnchorA = new Vec2(riderparts.get("Arm").body.getLocalCenter().x, riderparts.get("Arm").body.getLocalCenter().y - riderparts.get("Arm").h/2);
        revoluteJointDef.localAnchorB = new Vec2(pointerFront.body.getLocalCenter());
        revoluteJointDef.collideConnected = false;
        revoluteJointDef.enableLimit = true;
        revoluteJointDef.upperAngle = (float)Math.toRadians(-50);
        revoluteJointDef.lowerAngle = (float) Math.toRadians(-60);
        jointDefMap.add(new Imp_Joint("Arm_to_Bike",CONSTANTS.world.createJoint(revoluteJointDef),true));

        revoluteJointDef.bodyA = riderparts.get("Leg").body;
        revoluteJointDef.bodyB = bikebody.body;
        revoluteJointDef.localAnchorA = new Vec2(riderparts.get("Leg").body.getLocalCenter().x,riderparts.get("Leg").body.getLocalCenter().y - riderparts.get("Leg").h/2);
        revoluteJointDef.localAnchorB = new Vec2(bikebody.body.getLocalCenter().x+bikebody.w /2, bikebody.body.getLocalCenter().y );
        revoluteJointDef.upperAngle = (float)Math.toRadians(-70);
        revoluteJointDef.lowerAngle = (float) Math.toRadians(-60);
        jointDefMap.add(new Imp_Joint("Leg_to_Bike",CONSTANTS.world.createJoint(revoluteJointDef),true));


        shapes.addAll(riderparts.values());
        sys = new ParticleSys(5f,10,0.1f,Color.orange,ShapeType.CIRCLE,5f,false);

    }

    private void Create_bike_backpack(Vec2 initialState){
        float x = initialState.x;
        float y = initialState.y;
        backpack = new BikeBackpack(new Vec2(x,y), new Vec2(0,0), 4,Color.yellow, 0.001f, 2f, null,8,2f ,1.5f );
        WeldJointDef weldJointDef = new WeldJointDef();
        weldJointDef.bodyA = backpack.backpackList.get("Right").body;
        weldJointDef.bodyB = riderparts.get("Body").body;
        weldJointDef.localAnchorA.set(new Vec2( backpack.backpackList.get("Right").body.getLocalCenter()));
        weldJointDef.localAnchorB.set(new Vec2(riderparts.get("Body").body.getLocalCenter().x - riderparts.get("Body").w, riderparts.get("Body").body.getLocalCenter().y));
        jointDefMap.add(new Imp_Joint("Pack_to_Body",CONSTANTS.world.createJoint(weldJointDef),true));
        shapes.addAll(backpack.backpackList.values());
        shapes.addAll(backpack.waterList);
    }

    private void CreateTerrrains(Vec2 InitialState, float level_width, GameController gm){


        float x = InitialState.x;
        float y = InitialState.y;
        terrainUtility = new TerrainUtility();


        int num_of_terrains = gm.levels.get(gm.level).size();

        System.out.println("level is: "+ gm.level+ " number of terrains are " +num_of_terrains);
        int i = 1;
        for(Terrain.TerrainType  t: gm.levels.get(gm.level)){

            Vec2 position = new Vec2( level_width/(num_of_terrains+1)*i++, Terrain.TerrainType.Easy.getHeight()-0.5f);

            terrainUtility.add_to_TerrainList(new Terrain.TerrainShape(position.x,Color.WHITE,t));

       }
        //terrainUtility.add_to_TerrainList(new Terrain.TerrainShape(new Vec2( 20+level_width/num_of_terrains,y-2), 20,5f, Color.WHITE, Terrain.TerrainType.Steep));



    }

    public PhysicsEngine(){

        gm = new GameController(10);
        initialise_world();


    }
    public static void main(String[] args) throws Exception {
        final PhysicsEngine game = new PhysicsEngine();
        final BasicView view = new BasicView(game);
        JEasyFrame frame = new JEasyFrame(view, "Basic Physics Engine");
        frame.addKeyListener(new BasicKeyListener());
        view.addMouseMotionListener(new BasicMouseListener());
        game.startThread(view);
    }





    private void BackJoint(Bike body, Bike BWheel, Bike Engine, Bike shock){
        WeldJointDef w_joint = new WeldJointDef();
        w_joint.initialize(Engine.body, body.body, new Vec2(body.body.getWorldCenter().x, body.body.getWorldCenter().y + body.h));
        jointDefMap.add( new Imp_Joint("Engine_to_Bike",CONSTANTS.world.createJoint(w_joint),true));

        PrismaticJointDef prismaticJointDef = new PrismaticJointDef();
        prismaticJointDef.initialize(Engine.body,shock.body,Engine.body.getWorldCenter(),Engine.body.getWorldCenter());
        prismaticJointDef.localAnchorB.set(Engine.body.getLocalCenter().x, Engine.body.getLocalCenter().y - Engine.h/2);
        prismaticJointDef.enableLimit = true;
        prismaticJointDef.lowerTranslation = -0.2f;
        prismaticJointDef.upperTranslation = -0.3f;
        jointDefMap.add( new Imp_Joint("Shock_to_Engine", CONSTANTS.world.createJoint(prismaticJointDef), true));

        RevoluteJointDef revoluteJoint = new RevoluteJointDef();
        revoluteJoint.bodyA = BWheel.body;
        revoluteJoint.bodyB = shock.body;
        revoluteJoint.localAnchorA.set(BWheel.body.getLocalCenter().x, BWheel.body.getLocalCenter().y);
        revoluteJoint.localAnchorB.set(shock.body.getLocalCenter().x - shock.w/2 , shock.body.getLocalCenter().y - shock.h/2);//-2.3f*SCALE);//0.5f*SCALE,0.475f*SCALE);
        revoluteJoint.collideConnected = false;
        jointDefMap.add(new Imp_Joint("BW_to_Shock",CONSTANTS.world.createJoint(revoluteJoint),true));


    }
    private void FrontJoint(Bike body, Bike FWheel, Bike shock, Bike FrontJoint){

        WeldJointDef w_joint  = new WeldJointDef();
        w_joint.initialize(FrontJoint.body, body.body, new Vec2(body.body.getWorldCenter().x + body.w, body.body.getWorldCenter().y + body.h/2));
        jointDefMap.add(new Imp_Joint("FrontP_to_Bike",CONSTANTS.world.createJoint(w_joint),true));

        PrismaticJointDef prismaticJointDef = new PrismaticJointDef();
        prismaticJointDef.initialize(FrontJoint.body,shock.body,FrontJoint.body.getWorldCenter(),FrontJoint.body.getWorldCenter());
        prismaticJointDef.enableLimit = true;
        prismaticJointDef.lowerTranslation = -0.2f;
        prismaticJointDef.upperTranslation = -0.3f;
        prismaticJointDef.referenceAngle = 30;
        jointDefMap.add(new Imp_Joint("FrontP_to_Shock",CONSTANTS.world.createJoint(prismaticJointDef),true));
        RevoluteJointDef revoluteJoint = new RevoluteJointDef();
        revoluteJoint.bodyA = FWheel.body;
        revoluteJoint.bodyB = shock.body;
        revoluteJoint.localAnchorA.set(FWheel.body.getLocalCenter().x, FWheel.body.getLocalCenter().y);
        revoluteJoint.localAnchorB.set(shock.body.getLocalCenter().x , shock.body.getLocalCenter().y - shock.h/2);//-2.3f*SCALE);//0.5f*SCALE,0.475f*SCALE);
        revoluteJoint.collideConnected = false;
        jointDefMap.add(new Imp_Joint("FW_to_Shock",CONSTANTS.world.createJoint(revoluteJoint),true));
        DistanceJointDef distanceJointDef = new DistanceJointDef();
        distanceJointDef.bodyA = body.body;
        distanceJointDef.bodyB = shock.body;
        distanceJointDef.localAnchorA.set(body.body.getLocalCenter().x+ body.w, body.body.getLocalCenter().y);
        distanceJointDef.localAnchorB.set(shock.body.getLocalCenter().x, shock.body.getLocalCenter().y);
        distanceJointDef.length = 0.1f;
        distanceJointDef.dampingRatio = 2f;
        distanceJointDef.frequencyHz = 25f;

        jointDefMap.add(new Imp_Joint("Shock_to_Bike", CONSTANTS.world.createJoint(distanceJointDef),true));
    }
    private void startThread(final BasicView view) throws InterruptedException {
        final PhysicsEngine game=this;
        while (true) {
            game.update();
            view.repaint();

            try {
                Thread.sleep(CONSTANTS.DELAY);
            } catch (InterruptedException e) {
            }
        }
    }
    public void update() {

        check();
        int VELOCITY_ITERATIONS=CONSTANTS.NUM_EULER_UPDATES_PER_SCREEN_REFRESH;
        int POSITION_ITERATIONS=CONSTANTS.NUM_EULER_UPDATES_PER_SCREEN_REFRESH;
        clouds.updateClouds();
        sys.update();
        terrainUtility.update();


          for(Bike b : bikeparts){

              if(b.body.getUserData() == "FrontBikeWheel" && ((BikeWheels)b).TouchingTheGround) {
                  if (BasicKeyListener.isRotateLeftKeyPressed()) {
                      Vec2 newVec = b.body.getPosition();
                      newVec = new Vec2(newVec.x, newVec.y - 0.5f);
                      sys.updateLocation(newVec, new Vec2(newVec.x + 2f, newVec.y + 0.5f));

                  }
              }
              if(b.body.getUserData() == "BackBikeWheel" && ((BikeWheels)b).TouchingTheGround) {

                  if (BasicKeyListener.isRotateRightKeyPressed()){
                      Vec2 newVec = b.body.getPosition();
                      newVec = new Vec2(newVec.x, newVec.y - 0.5f);
                      sys.updateLocation( newVec, new Vec2(newVec.x -2f, newVec.y + 0.5f));

                  }

              }

          }
        UTIL.RemoveJoint(jointDefMap);
        if(!shapes.isEmpty()){
        for(Shapes s : shapes){
            s.update();
        }}
        if(!bikeparts.isEmpty()) {
            for (Bike b : bikeparts) {


                b.notificationOfNewTimestep();


            }
        }

        CONSTANTS.world.step(CONSTANTS.DELTA_T, VELOCITY_ITERATIONS, POSITION_ITERATIONS);
    }

    private void check(){

        if(!UTIL.Existance_Of_Joint("Arm_to_Bike",jointDefMap)){

            initialise_world();
            gm.reset();

        }
        else if(bikeparts.get(0).body.getPosition().x >= gm.LevelWidth()+30 ){

            gm.calculateScore(true, backpack.dropCount());
            gm.newGame();
            initialise_world();
            gm.reset();
        }
        if(!UTIL.Existance_Of_Joint("Pack_to_Body",jointDefMap)){
            gm.removeBonus();
        }


    }
    private void ADDEDJOINTS1(Bike b1, Bike b2){

        RevoluteJointDef joint = new RevoluteJointDef();

        joint.bodyA = b1.body;
        joint.bodyB = b2.body;
        joint.collideConnected = false;
        joint.localAnchorA=new Vec2(-1.25f, -.75f);
        joint.localAnchorB=new Vec2(0, -0.01f);
       // joint.enableLimit=true;
      //  joint.lowerAngle=(float)Math.toRadians(-10);
        jointDefMap.add(new Imp_Joint("FW" ,CONSTANTS.world.createJoint(joint),true));
    }
    private void ADDEDJOINTS2(Bike b1, Bike b2){
        RevoluteJointDef joint = new RevoluteJointDef();

        joint.bodyA = b1.body;
        joint.bodyB = b2.body;
        joint.collideConnected = false;
        joint.localAnchorA = new Vec2(1.25f,-.75f);
        joint.localAnchorB=new Vec2(0, -0.01f);
       // joint.enableLimit=true;
        //joint.lowerAngle=(float)Math.toRadians(-10);
        jointDefMap.add(new Imp_Joint("BW" ,CONSTANTS.world.createJoint(joint),true));

    }
    private void Apply_contact_rules(){

        CONSTANTS.world.setContactListener(new ContactListener() {

            @Override
            public void beginContact(Contact contact) {



                if((contact.getFixtureA().getBody().getUserData() == "Ground"&&
                        contact.getFixtureB().getBody().getUserData() == "backpack")|| (contact.getFixtureA().getBody().getUserData() == "backpack"&&
                        contact.getFixtureB().getBody().getUserData() == "Ground")) {
                    if(UTIL.Existance_Of_Joint("Pack_to_Body", jointDefMap)) {
                        //CONSTANTS.world.destroyJoint(jointDefMap.get("Pack_to_Body"));
                        UTIL.Return_Joint("Pack_to_Body", jointDefMap).exists = false;
                    }
                }

                //BODY AND FLOOR
                else if((contact.getFixtureA().getBody().getUserData() == "Ground"&&
                        contact.getFixtureB().getBody().getUserData() == "Head") || (contact.getFixtureA().getBody().getUserData() == "Head"&&
                        contact.getFixtureB().getBody().getUserData() == "Ground")) {
                    if(UTIL.Existance_Of_Joint("Arm_to_Bike", jointDefMap)) {
                        //CONSTANTS.world.destroyJoint(jointDefMap.get("Pack_to_Body"));
                        UTIL.Return_Joint("Arm_to_Bike", jointDefMap).exists = false;
                    }
                    if(UTIL.Existance_Of_Joint("Leg_to_Bike", jointDefMap)) {
                        //CONSTANTS.world.destroyJoint(jointDefMap.get("Pack_to_Body"));
                        UTIL.Return_Joint("Leg_to_Bike", jointDefMap).exists = false;
                    }
                }



                if((contact.getFixtureA().getBody().getUserData() == "Ground"&&
                        contact.getFixtureB().getBody().getUserData() == "BackBikeWheel")|| (contact.getFixtureA().getBody().getUserData() == "BackBikeWheel"&&
                        contact.getFixtureB().getBody().getUserData() == "Ground")){



                    for (Bike b : bikeparts) {

                        if(b.body.getUserData() == "BackBikeWheel"){

                            ((BikeWheels) b).TouchingTheGround = true;

                        }

                    }

                }
                else{
                    for (Bike b : bikeparts) {

                        if(b.body.getUserData() == "BackBikeWheel"){

                            ((BikeWheels) b).TouchingTheGround = false;

                        }

                    }
                }



                if(contact.getFixtureA().getBody().getUserData() == "Ground"&&
                        contact.getFixtureB().getBody().getUserData().toString().split(",")[0].equals( "waterDrop"))
                {
                    backpack.addtoDropped(contact.getFixtureB().getBody().getUserData().toString());
                }

                if (contact.getFixtureA().getBody().getUserData().toString().split(",")[0].equals( "waterDrop")&&
                        contact.getFixtureB().getBody().getUserData() == "Ground") {

                    backpack.addtoDropped(contact.getFixtureA().getBody().getUserData().toString());



                }

                if((contact.getFixtureA().getBody().getUserData() == "Ground"&&
                        contact.getFixtureB().getBody().getUserData() == "FrontBikeWheel")|| (contact.getFixtureA().getBody().getUserData() == "FrontBikeWheel"&&
                        contact.getFixtureB().getBody().getUserData() == "Ground")){

                    for (Bike b : bikeparts) {

                        if(b.body.getUserData() == "FrontBikeWheel"){

                            ((BikeWheels) b).TouchingTheGround = true;

                        }

                    }
                }
                else{
                    for (Bike b : bikeparts) {

                        if(b.body.getUserData() == "FrontBikeWheel"){

                            ((BikeWheels) b).TouchingTheGround = false;

                        }

                    }
                }



            }

            @Override
            public void endContact(Contact contact) {

            }

            @Override
            public void preSolve(Contact contact, Manifold oldManifold) {

            }

            @Override
            public void postSolve(Contact arg0, ContactImpulse arg1) {
                // TODO Auto-generated method stub
            }

        });
    }
}

