package BalanceBikeGame;

import org.jbox2d.dynamics.joints.Joint;

public class Imp_Joint {

    public String Name;
    public boolean exists;
    public Joint joint;

    public Imp_Joint(String Name, Joint joint,boolean exists){
        this.Name = Name;
        this.exists = exists;
        this.joint = joint;

    }

}
