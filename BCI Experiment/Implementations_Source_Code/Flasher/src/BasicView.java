import javax.swing.JComponent;
import java.awt.*;

public class BasicView extends JComponent{

    FlasherMain flasher;
    public BasicView(){




    }

    public void  setup(FlasherMain flasherMain){
        flasher = flasherMain;

    }

    @Override
    public void paintComponent(Graphics g0){


        Graphics2D g = (Graphics2D) g0;

        g.setColor( Color.black);

        g.fillRect(0,0,getWidth(), getHeight());

        if (flasher != null) {
            for (CustomRec b : flasher.Buttons) {
                if (b != null)
                    b.draw(g);

            }
        }
        repaint();
    }



}
