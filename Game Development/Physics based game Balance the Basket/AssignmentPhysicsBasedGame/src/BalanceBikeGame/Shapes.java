package BalanceBikeGame;

import jdk.nashorn.internal.ir.annotations.Reference;
import org.jbox2d.collision.shapes.CircleShape;
import org.jbox2d.collision.shapes.PolygonShape;
import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.*;

import javax.rmi.CORBA.Util;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Path2D;

public abstract class Shapes {

    public  int SCREEN_RADIUS;
    public  float ratioOfScreenScaleToWorldScale;
    public  float rollingFriction,mass;
    public  Color color;
    protected Body body;
    public float radius;
    public float h,w;
    private boolean IS_Circle;
    private Path2D.Float polygonPath;
    PolygonShape shape;
    public static CircleShape circleShape;

    /**
     * Polygon maker
     * @param position
     * @param velocity
     * @param radius
     * @param color
     * @param mass
     * @param rollingFriction
     * @param polgonPath
     * @param numSides
     */
    public Shapes(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, Path2D.Float polgonPath, int numSides, String userData){
        SCREEN_RADIUS = 0;
        World w= CONSTANTS.world;
        BodyDef bodyDef = new BodyDef();
        bodyDef.type = BodyType.DYNAMIC;
        bodyDef.position.set(position);
        bodyDef.userData = userData;
        bodyDef.linearVelocity.set(velocity);
        this.radius = radius;
        bodyDef.angularDamping = 0.1f;
        this.body = w.createBody(bodyDef);
        shape = new PolygonShape();
        body.setAngularVelocity(0f);
        this.color = color;
        this.mass = mass;
        this.rollingFriction = rollingFriction;
        IS_Circle = false;
        if(polygonPath != null) {

            setPolygonPath(polygonPath,numSides);
        }
    }

    /**
     * Circle Maker
     * @param position
     * @param velocity
     * @param radius
     * @param color
     * @param mass
     * @param rollingFriction
     */
    public Shapes(Vec2 position, Vec2 velocity, float radius, Color color, float mass, float rollingFriction, String userData){


       // System.out.println(userData +"//"+ position+"//"+ velocity+"//"+ radius+"//"+color+"//"+mass+"//"+rollingFriction);
        World w= CONSTANTS.world;
        BodyDef bodyDef = new BodyDef();
        bodyDef.type = BodyType.DYNAMIC;
        bodyDef.position.set(position);
        bodyDef.linearVelocity.set(velocity);
        bodyDef.userData = userData;
        //bodyDef.angularDamping = 0.1f;
       // System.out.println(w.createBody(bodyDef));
        this.body = w.createBody(bodyDef);
        //System.out.println(body);

        circleShape = new CircleShape();
        circleShape.m_radius = radius;
        FixtureDef fixtureDef = new FixtureDef();
        fixtureDef.shape = circleShape;
        fixtureDef.density = (float) (mass/ (Math.pow(radius,2))*Math.PI );

        fixtureDef.friction = 9f;
        fixtureDef.restitution = 0.5f;
        body.createFixture(fixtureDef);
        this.rollingFriction=rollingFriction;
        this.mass=mass;
        this.SCREEN_RADIUS=(int)Math.max(CONSTANTS.convertWorldLengthToScreenLength(radius),1);

        this.color=color;
        ratioOfScreenScaleToWorldScale = 0;
        polygonPath = null;
        IS_Circle = true;
    }

    /**
     *
     * @param polygonPath
     * @param numSides
     */
    public void setPolygonPath(Path2D.Float polygonPath, int numSides) {

        Vec2[] vertices = UTIL.verticesOfPath2D(polygonPath, numSides);
        shape.set(vertices, numSides);
        FixtureDef fixtureDef = new FixtureDef();// This class is from Box2D
        fixtureDef.shape = shape;
        fixtureDef.density = (float) (mass/((float) numSides)/2f*(radius*radius)*Math.sin(2*Math.PI/numSides));
        fixtureDef.friction = 0.1f;// this is surface friction;
        fixtureDef.restitution = 0.5f;
        body.createFixture(fixtureDef);
        this.ratioOfScreenScaleToWorldScale = CONSTANTS.convertWorldLengthToScreenLength(1);

        this.polygonPath = polygonPath;

    }

    /**
     *
     * @param g
     */
    public void draw(Graphics2D g){
        if(IS_Circle){
            int x = CONSTANTS.convertWorldXtoScreenX(body.getPosition().x);
            int y = CONSTANTS.convertWorldYtoScreenY(body.getPosition().y);
            g.setColor(color);
            g.fillOval(x - SCREEN_RADIUS, y - SCREEN_RADIUS, 2 * SCREEN_RADIUS, 2 * SCREEN_RADIUS);
            //int x2 = (int) (x + 30 *  Math.cos(body.getAngle() * 180/Math.PI));
           // int y2 = (int) (y + 30 *Math.sin(body.getAngle()* 180/Math.PI));
           // g.setColor(Color.black);
           // g.drawLine(x, y, x2, y2);
        }
        else if(!IS_Circle) {
            g.setColor(color);
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
    public abstract void update();


}
