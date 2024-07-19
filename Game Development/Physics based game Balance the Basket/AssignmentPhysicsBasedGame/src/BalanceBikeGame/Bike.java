package BalanceBikeGame;
import org.jbox2d.collision.shapes.CircleShape;
import org.jbox2d.collision.shapes.PolygonShape;
import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.*;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Path2D;
import java.awt.geom.PathIterator;

import static org.jbox2d.particle.ParticleType.b2_waterParticle;

public abstract class Bike {
    public  int SCREEN_RADIUS;
    public  float ratioOfScreenScaleToWorldScale;
    public  float rollingFriction,mass;
    public  Color col;
    protected  Body body;
    public float radius;
    public float h,w;
    private boolean IS_Circle;
    private Path2D.Float polygonPath;
    PolygonShape shape;
    public static  float MAGNITUDE_OF_ENGINE_THRUST_FORCE = 120f;
    public static float MAGNITUDE_OF_BODY_THRUST_FORCE =100f;
    public static CircleShape circleShape;
    public Bike(float sx, float sy, float vx, float vy, float radius, Color col, float mass, float rollingFriction, Path2D.Float polygonPath, int numSides){
        SCREEN_RADIUS = 0;
        World w= CONSTANTS.world; // a Box2D object
        BodyDef bodyDef = new BodyDef();  // a Box2D object
        bodyDef.type = BodyType.DYNAMIC; // this says the physics engine is to move it automatically
        bodyDef.position.set(sx, sy);
        bodyDef.linearVelocity.set(vx, vy);
        bodyDef.userData = "BikePart";
        this.radius = radius;
        bodyDef.angularDamping = 0.1f;
        this.body = w.createBody(bodyDef);
        shape = new PolygonShape();
        body.setAngularVelocity(0f);
        this.col = col;
        this.mass = mass;
        this.rollingFriction = rollingFriction;
        IS_Circle = false;
        if(polygonPath != null) {

            setPolygonPath(polygonPath,numSides);
        }
    }

    public void setPolygonPath(Path2D.Float polygonPath, int numSides) {

        Vec2[] vertices = verticesOfPath2D(polygonPath, numSides);
        shape.set(vertices, numSides);
        FixtureDef fixtureDef = new FixtureDef();// This class is from Box2D
        fixtureDef.shape = shape;
        fixtureDef.density = (float) (mass/((float) numSides)/2f*(radius*radius)*Math.sin(2*Math.PI/numSides));
        fixtureDef.friction = 0.1f;// this is surface friction;
        fixtureDef.restitution = 0.5f;
        body.createFixture(fixtureDef);
        this.ratioOfScreenScaleToWorldScale = CONSTANTS.convertWorldLengthToScreenLength(1);
        //System.out.println("Screenradius="+ratioOfScreenScaleToWorldScale);
        this.polygonPath = polygonPath;

    }

    public Bike(float sx, float sy, float vx, float vy, float radius, Color col, float mass, float rollingFriction ){
        World w= CONSTANTS.world; // a Box2D object
        BodyDef bodyDef = new BodyDef();  // a Box2D object
        bodyDef.type = BodyType.DYNAMIC; // this says the physics engine is to move it automatically
        bodyDef.position.set(sx, sy);
        bodyDef.linearVelocity.set(vx, vy);
        bodyDef.userData = "BikePart";
        //bodyDef.angularDamping = 0.1f;
        this.body = w.createBody(bodyDef);
        circleShape = new CircleShape();
        circleShape.m_radius = radius;
        FixtureDef fixtureDef = new FixtureDef();// This class is from Box2D
        fixtureDef.shape = circleShape;
        fixtureDef.density = (float) (mass/ (Math.pow(radius,2))*Math.PI );
        fixtureDef.friction = 9f;// this is surface friction;
        fixtureDef.restitution = 0.5f;
        body.createFixture(fixtureDef);
        this.rollingFriction=rollingFriction;
        this.mass=mass;
        this.SCREEN_RADIUS=(int)Math.max(CONSTANTS.convertWorldLengthToScreenLength(radius),1);
        //System.out.println("Screenradius="+ratioOfScreenScaleToWorldScale);
        this.col=col;
        ratioOfScreenScaleToWorldScale = 0;
        polygonPath = null;
        IS_Circle = true;

    }
    public void draw(Graphics2D g){

        if(IS_Circle){
            int x = CONSTANTS.convertWorldXtoScreenX(body.getPosition().x);
            int y = CONSTANTS.convertWorldYtoScreenY(body.getPosition().y);
            g.setColor(col);
            g.fillOval(x - SCREEN_RADIUS, y - SCREEN_RADIUS, 2 * SCREEN_RADIUS, 2 * SCREEN_RADIUS);
            int x2 = (int) (x + 30 *  Math.cos(body.getAngle() * 180/Math.PI));
            int y2 = (int) (y + 30 *Math.sin(body.getAngle()* 180/Math.PI));
            g.setColor(Color.black);
            g.drawLine(x, y, x2, y2);
        }
        else if(!IS_Circle) {
            g.setColor(col);
            Vec2 position = body.getPosition();
            float angle = body.getAngle();
            AffineTransform af = new AffineTransform();
            af.translate(CONSTANTS.convertWorldXtoScreenX(position.x), CONSTANTS.convertWorldYtoScreenY(position.y));
            af.scale(ratioOfScreenScaleToWorldScale, -ratioOfScreenScaleToWorldScale);// there is a minus in here because screenworld is flipped upsidedown compared to physics world
            af.rotate(angle);
            Path2D.Float p = new Path2D.Float(polygonPath, af);
            g.fill(p);
        }
    }
    public static boolean areAllTrue(boolean[] array) {
        for (boolean b : array) if (!b) return false;
        return true;
    }


    public abstract void notificationOfNewTimestep();
    // Vec2 vertices of Path2D
    public static Vec2[] verticesOfPath2D(Path2D.Float p, int n) {
        Vec2[] result = new Vec2[n];
        float[] values = new float[6];
        PathIterator pi = p.getPathIterator(null);
        int i = 0;
        while (!pi.isDone() && i < n) {
            int type = pi.currentSegment(values);
            if (type == PathIterator.SEG_LINETO) {
                result[i++] = new Vec2(values[0], values[1]);
            }
            pi.next();
        }
        return result;
    }
}
