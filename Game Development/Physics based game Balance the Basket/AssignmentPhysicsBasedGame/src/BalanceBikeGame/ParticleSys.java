package BalanceBikeGame;


import org.jbox2d.collision.shapes.ShapeType;
import org.jbox2d.common.MathUtils;
import org.jbox2d.common.Vec2;

import java.awt.*;
import java.awt.geom.Path2D;
import java.sql.Time;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Timer;

public class ParticleSys {
    public class Particlebits extends Shapes{
        private boolean FirstTime;
        private Vec2 destination;
        private Vec2 initialpos;
        public Vec2 Finishpos;
        private float TimeAlive;
        private float currentTimer;
        private boolean repeat;
        public boolean alive;
        public Particlebits(Vec2 position,  Vec2 velocity, float radius, Color color, float mass, float rollingFriction, int numSides,float width, float height,float TimeAlive, String userData,boolean repeat, Vec2 finishpos) {
            super(position, velocity, radius, color, mass, rollingFriction, null, numSides, userData);
            this.setPolygonPath(make_rec(width, height), 4);
          //  this.body.setGravityScale(0f);
            this.initialpos = position;
            destination = new Vec2(0,0);
            FirstTime = true;
            this.TimeAlive = TimeAlive;
            currentTimer = 0;
            this.repeat = repeat;
            this.alive = true;
            body.getFixtureList().getFilterData().maskBits = 0*0002;
            this.Finishpos = finishpos;
        }

        public Particlebits(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction,float TimeAlive, String userData, boolean repeat, Vec2 finishpos) {
            super(position, velocity, radius, color, mass, rollingFriction, userData);
            this.repeat = repeat;
           // this.body.setGravityScale(0f);
            FirstTime = true;
            this.initialpos = position;
            destination = new Vec2(0,0);
            this.TimeAlive = TimeAlive;
            currentTimer = 0;
            this.alive = true;
            body.getFixtureList().getFilterData().maskBits = 0*0002;
            this.Finishpos = finishpos;
        }

        public Vec2 getDestination() {
            return destination;
        }

        public void setDestination(Vec2 destination) {
            this.destination = destination;
        }
        public void generatenewDestination(float cone_radius, Vec2 defaultdesitination){

            Vec2 pos_ref = body.getPosition();
            pos_ref = (pos_ref.add(defaultdesitination.mul(-1)));
            float Actual_distance = (float) Math.sqrt(Math.pow(pos_ref.x, 2) + Math.pow(pos_ref.y, 2));
            currentTimer += 0.1f;
            if( FirstTime  || repeat ) {
                if ((currentTimer >= TimeAlive && TimeAlive > 0) ||Math.abs(Actual_distance ) < 0.1f ) {
                    currentTimer = 0;
                    FirstTime = false;
                    Random random = new Random();
                    float nextradius = (-cone_radius) + random.nextFloat() * (cone_radius + cone_radius);

                    body.setTransform(initialpos, body.getAngle());
                    Vec2 initialpos_ref = initialpos;
                    initialpos_ref = (initialpos_ref.add(defaultdesitination.mul(-1)));
                    float lenghtAB = (float) Math.sqrt(Math.pow(initialpos_ref.x, 2) + Math.pow(initialpos_ref.y, 2));
                    initialpos_ref = initialpos;
                    float lengthBC = Math.abs(nextradius);
                    float lenghAC = (float) Math.sqrt(Math.pow(lenghtAB, 2) + Math.pow(lengthBC, 2));


                    double a = ((lenghAC * lenghAC - lengthBC * lengthBC) + lenghtAB * lenghtAB) / (2 * lenghtAB);
                    double h = Math.sqrt(lenghAC * lenghAC - a * a);

                    Vec2 temp = new Vec2((float) (initialpos_ref.x + a * (defaultdesitination.x - initialpos_ref.x)) / lenghtAB, (float) (initialpos_ref.y + a * (defaultdesitination.y - initialpos_ref.y) / lenghtAB));

                    destination.set(new Vec2((float) (temp.x + (h * (nextradius / nextradius)) * (defaultdesitination.x - initialpos_ref.x) / lenghtAB), (float) (temp.y - (h * (nextradius / nextradius)) * (defaultdesitination.y - initialpos_ref.y) / lenghtAB)));

                    body.applyForceToCenter(destination.mul(0.1f));
                }
            }
            else if((currentTimer >= TimeAlive && TimeAlive > 0)){
               alive = false;
            }
        }

        @Override
        public void update() {




        }
        public Path2D.Float make_rec(float width, float height){
            Path2D.Float p = new Path2D.Float();
            p.moveTo(-0.5*width, -0.5*height);
            p.lineTo(-0.5* width, 0.5* height);
            p.lineTo(0.5*width, 0.5*height);
            p.lineTo(0.5*width, -0.5*height);
            p.lineTo(-0.5*width, -0.5 *height);
            p.closePath();
            return p;

        }
    }


    private Vec2 position, velocity, finishPos;
    private float cone_radius, particleradius;
    private int Num_particles, numsides;
    List<Particlebits> particlebitsList;
    private Color color;
    private ShapeType shapeType;
    private float TimeAlive;
    private  boolean repeat;
    public ParticleSys(Vec2 position, Vec2 finishPos, Vec2 velocity, float cone_radius , int Num_particles, float particle_radius  ,Color color, ShapeType shapeType, boolean repeat){

        this.repeat = repeat;
        this.color = color;
        this.position =position;
        this.velocity = velocity;
        this.cone_radius = cone_radius;
        this.Num_particles = Num_particles;
        this.particleradius =  particle_radius;
        particlebitsList = new ArrayList<>();
        this.shapeType = shapeType;
        this.finishPos = finishPos;
        this.TimeAlive = 0;

    }
    public ParticleSys(Vec2 position, Vec2 finishPos, Vec2 velocity, float cone_radius , int Num_particles, float particle_radius  ,Color color, ShapeType shapeType, float TimeAlive, boolean repeat){
        this.color = color;
        this.position =position;
        this.velocity = velocity;
        this.cone_radius = cone_radius;
        this.Num_particles = Num_particles;
        this.particleradius =  particle_radius;
        particlebitsList = new ArrayList<>();
        this.shapeType = shapeType;
        this.finishPos = finishPos;
        this.TimeAlive = TimeAlive;
        this.repeat = repeat;

    }


    public ParticleSys( float cone_radius , int Num_particles, float particle_radius  ,Color color, ShapeType shapeType, float TimeAlive, boolean repeat){

        this.color = color;
        this.repeat = repeat;
        this.cone_radius = cone_radius;
        this.Num_particles = Num_particles;
        this.particleradius =  particle_radius;
        particlebitsList = new ArrayList<>();
        this.shapeType = shapeType;
        this.TimeAlive = 0.5f;
        this.velocity = new Vec2(0,0);



    }
    public void updateLocation(Vec2 position, Vec2 finishPos){

        for (int i  = 0; i < Num_particles; i++){

            switch (shapeType) {
                case CIRCLE:

                  // System.out.println(position +"//"+  velocity +"//"+  particleradius + "//"+  color + "//"+ 0.001f + "//"+0f +"//"+ TimeAlive +"//"+"ParticleSystem" +"//"+ repeat );



                    particlebitsList.add( new  Particlebits( position,  velocity,  particleradius,  color,  0.001f, 0f, TimeAlive,"ParticleSystem", repeat, finishPos));
                    break;
                case POLYGON:

                    particlebitsList.add( new Particlebits(position, velocity,particleradius,color,0.001f,0f,4,particleradius, particleradius,TimeAlive,"ParticleSystem",repeat,finishPos));
                    break;

            }


        }

        this.finishPos = finishPos;

    }
    private void setupParticles(){





    }

    public void update(){

        if(particlebitsList.size() !=0) {
            for (Particlebits b : particlebitsList) {

                if(b.alive) {
                    b.generatenewDestination(cone_radius, b.Finishpos);
                    b.update();
                }
            }
        }




    }
    public void draw(Graphics2D g){

        if(particlebitsList.size() !=0) {
            for (Particlebits b : particlebitsList) {
                if(b.alive) {
                    b.draw(g);
                }

            }


        }
    }







}
