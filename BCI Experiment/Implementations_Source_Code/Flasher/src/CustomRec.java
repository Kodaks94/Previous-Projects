import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class CustomRec implements ActionListener {

    CONSTANTS.RecID id;
    Vec2D pos;
    Color color;
    int width = CONSTANTS.size.x / 2, height = CONSTANTS.size.y/2;
    private boolean isFlashing=  true;
    private final Timer timer;
    private  int HZ = 60;
    private long next_sec = System.currentTimeMillis() + 1000;
    private int LS_frame = 0 , CS_frame = 0;
    public CustomRec(CONSTANTS.RecID id, int HZ){
        this.id = id;
        this.HZ = HZ;
        pos = new Vec2D(0,0);
        setupRec();
        this.timer =  new Timer(1000/HZ,  this);
        timer.start();
    }

    private void setupRec(){

        switch (id) {
            case one:
                pos.set(0,0);
                color = Color.RED;
                break;
            case two:
                pos.set(width +10, 0);
                color = Color.RED;
                break;
            case three:
                pos.set(0, height+10);
                color = Color.RED;
                break;
            case four:
                pos.set(width+ 10, height +10);
                color = Color.RED;
                break;
        }

    }

    public void draw(Graphics2D g){


        long current_T = System.currentTimeMillis();

        if(current_T > next_sec){

            next_sec += 1000;
            LS_frame = CS_frame;
            CS_frame = 0;
        }

        CS_frame++;

        g.setColor(color);
        if (isFlashing){
            g.fillRect(pos.x, pos.y, width, height);
            if(LS_frame != 0) {

                write_in_csv((current_T / 1000) % 60, LS_frame);
            }
            }

        else clear(g);

    }

    private void write_in_csv(long T, int fps){

        try{

            FileWriter CSV_file = new FileWriter("Analyser/Java_FrameFSP_Analyser.csv", true);

            CSV_file.append(T + "," + fps);
            CSV_file.append("\n");

            CSV_file.flush();
            CSV_file.close();


        } catch (IOException e) {

            System.out.println(e.fillInStackTrace());
        }


    }
    public void clear(Graphics2D g){
        g.clearRect(pos.x,pos.y,width,height);

    }

    @Override
    public void actionPerformed(ActionEvent e) {
        isFlashing = !isFlashing;
    }
}
