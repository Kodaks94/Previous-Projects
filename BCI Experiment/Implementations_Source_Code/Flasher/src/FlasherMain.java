import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionListener;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FlasherMain extends JFrame {



    List<CustomRec> Buttons;
    public Component comp;
    int StartingHZ = 0;
    int num = 0;
    public FlasherMain(Component comp, int num, int StartingHZ) {
        Buttons = new ArrayList<>();
        this.comp = comp;
        this.num = num;
        this.StartingHZ = StartingHZ;
            setup();

    }

    private void setup() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setVisible(true);
        getContentPane().add(BorderLayout.CENTER, comp);
        setBackground(Color.BLACK);
        setSize(CONSTANTS.size.x+20, CONSTANTS.size.y+50);
        setLocationRelativeTo(null);
        if(num == 0)
         num = Check();

        if(num != 0){

            add_squares(num);


        }
        //Buttons.add(new CustomRec(CONSTANTS.RecID.one, 120));
        repaint();

    }

    private void add_squares(int n){

        int st = CONSTANTS.STARTING_HZ;

        if(StartingHZ != 0) st = StartingHZ;
        for (int i = 0; i < n; i++){

            System.out.println(i);
            Buttons.add(new CustomRec(CONSTANTS.RecID.values()[i],st+ (5 * i) ));

        }


    }

    private int Check(){

        try{

            FileReader fileReader = new FileReader("instructions.txt");
           char [] a = new char [4];


            fileReader.read(a);


            return Character.getNumericValue(a[0]);


    } catch (FileNotFoundException e) {
            return 0;
        } catch (IOException e1) {
            return 0;
        }

    }

        public static void main(String[] args){
        int Num = 0;
        int StartingHZ= 0;
        BasicView basicView = new BasicView();
        if (args.length > 0){

            Num = Integer.valueOf(args[0]);
            StartingHZ = Integer.valueOf(args[1]);

            FlasherMain flasherMain = new FlasherMain(basicView, Num, StartingHZ);
            basicView.setup(flasherMain);

        }
        else {
            FlasherMain flasherMain = new FlasherMain(basicView, 0, 0);
            basicView.setup(flasherMain);
        }



    }


}

