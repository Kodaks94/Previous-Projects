package BalanceBikeGame;

import org.jbox2d.dynamics.World;

import java.awt.*;

public class  CONSTANTS {
    // frame dimensions
    public static final int SCREEN_HEIGHT = 680;
    public static final int SCREEN_WIDTH = 640;
    public static final Dimension FRAME_SIZE = new Dimension(
            SCREEN_WIDTH, SCREEN_HEIGHT);
    public static final float WORLD_WIDTH=10;//metres
    public static final float WORLD_HEIGHT=SCREEN_HEIGHT*(WORLD_WIDTH/SCREEN_WIDTH);// meters - keeps world dimensions in same aspect ratio as screen dimensions, so that circles get transformed into circles as opposed to ovals
    public static final float GRAVITY=9.8f;
    public static final boolean ALLOW_MOUSE_POINTER_TO_DRAG_BODIES_ON_SCREEN=true;// There's a load of code in basic mouse listener to process this, if you set it to true

    public static World world; // Box2D container for all bodies and barriers

    // sleep time between two drawn frames in milliseconds
    public static final int DELAY = 20;
    public static final int NUM_EULER_UPDATES_PER_SCREEN_REFRESH=10;
    // estimate for time between two frames in seconds
    public static final float DELTA_T = DELAY / 1000.0f;


    public static int convertWorldXtoScreenX(float worldX) {
        return (int) (worldX/WORLD_WIDTH*SCREEN_WIDTH);
    }
    public static int convertWorldYtoScreenY(float worldY) {
        // minus sign in here is because screen coordinates are upside down.
        return (int) (SCREEN_HEIGHT-(worldY/WORLD_HEIGHT*SCREEN_HEIGHT));
    }
    public static float convertWorldLengthToScreenLength(float worldLength) {
        return (worldLength/WORLD_WIDTH*SCREEN_WIDTH);
    }
    public static float convertScreenXtoWorldX(int screenX) {
        return screenX*WORLD_WIDTH/SCREEN_WIDTH;
    }
    public static float convertScreenYtoWorldY(int screenY) {
        return (SCREEN_HEIGHT-screenY)*WORLD_HEIGHT/SCREEN_HEIGHT;
    }

}
