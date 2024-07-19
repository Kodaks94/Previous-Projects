package gui;

import assign.Deck;
import assign.Game;
import socket.PlayerClient;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;
import java.awt.event.ActionListener;
import java.util.Arrays;

/**
 * Created by mpishe on 22/11/2017.
 */
public class FrameControl extends JFrame{
    JPanel Stats = new JPanel(), choose = new JPanel();
    Controls mainAction;

    Game game;
    //SERVER VALUES
    public static boolean MultiplayerSelected = false;
    public static int MultiplayerSelectedint = 0;
    public PlayerClient pc;
    public int RoundNumber = 0;
    public int[] Prices;
    public int[] Cards;
    public int[] Shares;
    public int playerID = 0;
    public int CashNumber;
    ////
    ///Waiting Labels
    public JLabel waiting;
    public JTextArea status;
    public String actions;
    public JButton wait;
    // Finishing
    public JLabel results;
    ///buttons and labels for frame
    public JPanel loginpanel = new JPanel(), TradePanel = new JPanel(), votepanel = new JPanel(), WaitingPanel = new JPanel(), ResultsPanel= new JPanel();
    public JLabel Phase, Cash, Round, ShareLabel;
    public JButton login, loginMultiplayer ;
    public JButton SkipVote;
    public JLabel NO, YES;
    public JCheckBox AppleNoCheck, BPNoCheck, CiscoNoCheck,DellNoCheck, EricssonNoCheck;
    public JLabel ApplePrice, BPPrice, CiscoPrice, DellPrice, EricssonPrice;
    public JButton BuyApple, BuyBP , BuyCisco , BuyDell, BuyEricsson, Skip, Accept;
    public JTextField AppleText, BPText, CiscoText, DellText, EricssonText;
    public JCheckBox AppleCheck, BPCheck, CiscoCheck, DellCheck, EricssonCheck;
    public JLabel AppleCard, BPCard, CiscoCard, DellCard, EricssonCard;
    public int TradeCounter = 0;
    public  int PhaseCounter = 0;
    public FrameControl(String title, Game game){
        super(title);
        mainAction = new Controls(this);
        setSize(1500,1000);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Phase = new JLabel("");
        Round =  new JLabel("Round: "+ RoundNumber);
        Cash = new JLabel("Cash: ");
        ShareLabel = new JLabel("");
        ShareLabel.setBounds(800,800,100,100);
        Login();
        TradeFrame();
        VoteFrame();
        WaitingFrame();
        Result();
        Stats.add(Round);
        Stats.add(Cash);
        Stats.add(ShareLabel);
        if(PhaseCounter == 0){
            setpanel(loginpanel);

        }



    }
    public void Result(){
        results = new JLabel("");
        ResultsPanel.add(results);

    }
    public void NetworkUpdate(){
        RoundNumber = pc.GetRound();
        Prices = pc.getPrices();
        Cards = pc.getCards();
        CashNumber = pc.GetCash(playerID);
        Shares = pc.getShare(playerID);
        playerID = pc.playerID;
    }
    public void setpanel(JPanel panel){
        JPanel contentPane = (JPanel) this.getContentPane();
        contentPane.removeAll();
        if(PhaseCounter == 1){
            NetworkUpdate();
            Buysetup();
        }
        else if(PhaseCounter == 2){
            Sellsetup();
        }
        Stats.setLayout(new FlowLayout());
        choose.removeAll();
        choose.add(panel);
        if(PhaseCounter != 0 ) {
            contentPane.add(Stats);
        }
        contentPane.add(choose);
        contentPane.revalidate();
        contentPane.repaint();
    }
    public void WaitingFrame(){
        waiting = new JLabel("Waiting for other Players");
        wait = new JButton("Check to proceed to next Round");
        wait.addActionListener(mainAction);
        status =  new JTextArea(actions,20,20);
        WaitingPanel.add(waiting);
        //WaitingPanel.add(status);
        WaitingPanel.add(wait);
    }
    public void TradeFrame(){
        Phase = new JLabel("Phase 1");
        AppleText = new JTextField("Amount");
        BPText = new JTextField("Amount");
        CiscoText=new JTextField("Amount");
        DellText=new JTextField("Amount");
        EricssonText =new JTextField("Amount");
        ApplePrice = new JLabel("Apple Price");
        BPPrice = new JLabel("BP Price");
        CiscoPrice = new JLabel("Cisco Price");
        DellPrice = new JLabel("Dell Price");
        EricssonPrice = new JLabel("Erricson Price");
        Skip = new JButton("Skip!");
        BuyApple = new JButton("Buy Apple Stock");
        BuyBP = new JButton("Buy BP Stock");
        BuyCisco = new JButton("Buy Cisco Stock");
        BuyDell = new JButton("Buy Dell Stock");
        BuyEricsson = new JButton("Buy Ericsson Stock");
        Skip.addActionListener(mainAction);
        BuyApple.addActionListener(mainAction);
        BuyBP.addActionListener(mainAction);
        BuyCisco.addActionListener(mainAction);
        BuyDell.addActionListener(mainAction);
        BuyEricsson.addActionListener(mainAction);

        TradePanel.setLayout(new GridBagLayout());
        GridBagConstraints gc = new GridBagConstraints();
        //Labels
        TradePanel.add(Round,gc);
        gc.insets = new Insets(2,2,2,2);
        gc.gridx = 0;
        gc.gridy = 1;
        TradePanel.add(Phase, gc);
        gc.gridx++;
        TradePanel.add(ApplePrice,gc);
        gc.gridx++;
        TradePanel.add(BPPrice, gc);
        gc.gridx++;
        TradePanel.add(CiscoPrice, gc);
        gc.gridx++;
        TradePanel.add(DellPrice, gc);
        gc.gridx++;
        TradePanel.add(EricssonPrice, gc);
        ///Buttons
        gc.gridx = 1;
        gc.gridy++;
        gc.fill = GridBagConstraints.HORIZONTAL;
        TradePanel.add(BuyApple,gc);
        gc.gridx++;
        TradePanel.add(BuyBP, gc);
        gc.gridx++;
        TradePanel.add(BuyCisco, gc);
        gc.gridx++;
        TradePanel.add(BuyDell, gc);
        gc.gridx++;
        TradePanel.add(BuyEricsson, gc);
        gc.gridx++;
        TradePanel.add(Skip, gc);
        ///TextFields
        gc.gridx = 1;
        gc.gridy++;
        gc.fill = GridBagConstraints.HORIZONTAL;
        TradePanel.add(AppleText,gc);
        gc.gridx++;
        TradePanel.add(BPText,gc);
        gc.gridx++;
        TradePanel.add(CiscoText,gc);
        gc.gridx++;
        TradePanel.add(DellText,gc);
        gc.gridx++;
        TradePanel.add(EricssonText,gc);
        TradePanel.repaint();
    }
    public void Sellsetup(){
        Phase.setText("Phase 2: Sell Stock Phase");
        BuyApple.setText("Sell Apple Stock");
        BuyBP.setText("Sell BP Stock");
        BuyCisco.setText("Sell Cisco Stock");
        BuyDell.setText("Sell Dell Stock");
        BuyEricsson.setText("Sell Ericsson Stock");
    }
    public void Buysetup(){
        Phase.setText("Phase 1: Buy Stock Phase");
        BuyApple.setText("Buy Apple Stock");
        BuyBP.setText("Buy BP Stock");
        BuyCisco.setText("Buy Cisco Stock");
        BuyDell.setText("Buy Dell Stock");
        BuyEricsson.setText("Buy Ericsson Stock");
    }

    public void VoteFrame(){
        SkipVote = new JButton("Skip");
        SkipVote.addActionListener(mainAction);
        AppleCheck = new JCheckBox("Apple Stock effect");
        AppleNoCheck = new JCheckBox("Apple Stock effect");
        AppleCard = new JLabel("Apple Card");
        BPCheck =  new JCheckBox("BP Stock effect");
        BPNoCheck =  new JCheckBox("BP Stock effect");
        BPCard = new JLabel("BP Card");
        CiscoCard =  new JLabel("Cisco Card");
        CiscoCheck = new JCheckBox("Cisco Stock effect");
        CiscoNoCheck = new JCheckBox("Cisco Stock effect");
        DellCard = new JLabel("Dell Card");
        DellCheck = new JCheckBox("Dell Stock effect");
        DellNoCheck = new JCheckBox("Dell Stock effect");
        EricssonCard = new JLabel("Ericsson Card");
        EricssonCheck =  new JCheckBox("Ericsson Stock effect");
        EricssonNoCheck =  new JCheckBox("Ericsson Stock effect");
        YES = new JLabel("Yes");
        NO = new JLabel("NO");
        Accept = new JButton("Accept");
        Accept.addActionListener(mainAction);
        votepanel.setLayout(new GridBagLayout());
        GridBagConstraints gc = new GridBagConstraints();
        gc.insets = new Insets(2,2,2,2);
        votepanel.add(Round,gc);
        gc.gridx = 0;
        gc.gridy = 1;
       // votepanel.add(Phase, gc);
        gc.gridx+=2;
        votepanel.add(AppleCard,gc);
        gc.gridx++;
        votepanel.add(BPCard, gc);
        gc.gridx++;
        votepanel.add(CiscoCard, gc);
        gc.gridx++;
        votepanel.add(DellCard, gc);
        gc.gridx++;
        votepanel.add(EricssonCard, gc);
        ///Buttons
        gc.gridx = 1;
        gc.gridy++;
        gc.fill = GridBagConstraints.HORIZONTAL;
        votepanel.add(YES, gc);
        gc.gridx++;
        votepanel.add(AppleCheck,gc);
        gc.gridx++;
        votepanel.add(BPCheck, gc);
        gc.gridx++;
        votepanel.add(CiscoCheck, gc);
        gc.gridx++;
        votepanel.add(DellCheck, gc);
        gc.gridx++;
        votepanel.add(EricssonCheck, gc);
        gc.gridx =1;
        gc.gridy++;
        votepanel.add(NO,gc);
        gc.gridx++;
        votepanel.add(AppleNoCheck, gc);
        gc.gridx++;
        votepanel.add(BPNoCheck, gc);
        gc.gridx++;
        votepanel.add(CiscoNoCheck, gc);
        gc.gridx++;
        votepanel.add(DellNoCheck,gc);
        gc.gridx++;
        votepanel.add(EricssonNoCheck,gc);
        gc.gridx++;
        gc.gridy++;
        votepanel.add(Accept, gc);
        gc.gridy++;
        votepanel.add(SkipVote, gc);
        votepanel.repaint();
    }
    public void Login(){

        Phase = new JLabel("Stock Game Simulator:");
        login = new JButton("Login to play as singlePlayer");
        loginMultiplayer = new JButton("Login to play as Multiplayer");
        setLayout(new GridBagLayout());
        login.addActionListener(mainAction);
        loginMultiplayer.addActionListener(mainAction);
        GridBagConstraints gc = new GridBagConstraints();
        //Labels
        gc.insets = new Insets(2,2,2,2);
        gc.gridx = 0;
        gc.gridy = 0;
        loginpanel.add(Phase, gc);
        gc.gridx++;
        loginpanel.add(login, gc);
        gc.gridy ++;
        loginpanel.add(loginMultiplayer, gc);

    }
    public void Reset(){
        AppleCheck.setSelected(false);
        BPCheck.setSelected(false);
        CiscoCheck.setSelected(false);
        DellCheck.setSelected(false);
        EricssonCheck.setSelected(false);
        AppleNoCheck.setSelected(false);
        BPNoCheck.setSelected(false);
        CiscoNoCheck.setSelected(false);
        DellNoCheck.setSelected(false);
        EricssonNoCheck.setSelected(false);
        AppleText.setText("Amount");
        BPText.setText("Amount");
        CiscoText.setText("Amount");
        DellText.setText("Amount");
        EricssonText.setText("Amount");

    }
    public void Update(){


            NetworkUpdate();
            ApplePrice.setText("Apple Price: " + Prices[0]);
            BPPrice.setText("BP Price: " + Prices[1]);
            CiscoPrice.setText("Cisco Price: " + Prices[2]);
            DellPrice.setText("Cisco Price: " + Prices[3]);
            EricssonPrice.setText("Cisco Price: " + Prices[4]);
            AppleCard.setText("Apple Card: " + Cards[0]);
            BPCard.setText("BP Card: " + Cards[1]);
            CiscoCard.setText("Cisco Card: " + Cards[2]);
            DellCard.setText("Dell Card: " + Cards[3]);
            EricssonCard.setText("Ericsson Card: " + Cards[4]);
            Round.setText("Round: "+ RoundNumber);
            if (playerID != 0) {
                Cash.setText("Cash: " + CashNumber);
                int[] x = Shares;
                ShareLabel.setText(String.format("Shares: [A:%d,B:%d,C:%d,D:%d,E:%d]", x[0], x[1], x[2], x[3], x[4]));
                status.setText(actions);
            }
    }

    }

