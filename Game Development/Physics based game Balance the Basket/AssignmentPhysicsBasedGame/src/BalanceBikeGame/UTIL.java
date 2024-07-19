package BalanceBikeGame;

import org.jbox2d.common.Vec2;

import java.awt.geom.Path2D;
import java.awt.geom.PathIterator;
import java.util.List;

public class UTIL {
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

    public  static boolean Existance_Of_Joint(String name, List<Imp_Joint> joints){

        for (Imp_Joint j : joints){

            if(j.Name.equals(name) && j.exists){

                return true;
            }


        }
        return false;
    }

    public  static Imp_Joint Return_Joint(String name, List<Imp_Joint> joints){

        for (Imp_Joint j : joints){

            if(j.Name.equals(name) && j.exists){

                return j;
            }


        }
        return null;
    }
    public static void RemoveJoint(List<Imp_Joint> joints){

        for(Imp_Joint j : joints){

            if (!j.exists){

                CONSTANTS.world.destroyJoint(j.joint);
            }
        }



    }

}
