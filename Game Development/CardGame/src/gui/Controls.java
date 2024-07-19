package gui;

import assign.Stock;
import socket.PlayerClient;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by mpishe on 22/11/2017.
 */
public class Controls implements ActionListener {
    FrameControl frame;
    List<AI> bots = new ArrayList<>();

    Controls(FrameControl frame){
        this.frame = frame;

    }

    public void actionPerformed(ActionEvent e) {
        if (frame.RoundNumber <=7) {
            if (frame.PhaseCounter == 0) {
                Entering(e);
            } else if (frame.PhaseCounter == 1) {
                Buying(e);
            } else if (frame.PhaseCounter == 2) {
                Selling(e);
            } else if (frame.PhaseCounter == 3) {
                Voting(e);
            }
            else if (frame.PhaseCounter == 4){
                System.out.println(frame.RoundNumber);
                Checking(e);
            }
        }
    }
    private void Entering(ActionEvent e){
        if(e.getSource() == frame.loginMultiplayer){

            frame.pc = new PlayerClient();
            frame.playerID = frame.pc.playerID;

            if(frame.playerID == 0){
                frame.login.setText("Too many Players, Cannot Access!");
                frame.repaint();
            }
            else {
                frame.PhaseCounter = 1;
                frame.Update();
                frame.setpanel(frame.TradePanel);
                if(FrameControl.MultiplayerSelectedint != 1) {
                    frame.pc.ADDMUL();
                    FrameControl.MultiplayerSelectedint = 1;
                }
            }

        }
        if(e.getSource() == frame.login){
            frame.pc = new PlayerClient();
            frame.playerID = frame.pc.playerID;

            AI ai = new AI();
            AI ai2 = new AI();
            AI ai3 = new AI();
            ai.Update();
            ai2.Update();
            ai3.Update();
            bots.add(ai);
            bots.add(ai2);
            bots.add(ai3);

            frame.PhaseCounter = 1;
            frame.Update();
            frame.setpanel(frame.TradePanel);
        }

    }
    private void CashChcek(String a){
        if(a.equals("Not Enough Cash")){
            JOptionPane.showMessageDialog(frame,
                    "Check Cash, Keep in mind there is 3 pounds for each stock bought ",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE);
        }
        else{

            frame.Update();
        }
    }
    private void StockCheck(String a){
        if(a.equals("Not Enough Stock")){
            JOptionPane.showMessageDialog(frame,
                    "Check Stock, Keep in mind, you can not sell more than what you own! ",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE);
        }
        else {
            frame.TradeCounter++;
            frame.Update();
        }
    }
    private void Buying(ActionEvent e) {
        try {
            if (frame.TradeCounter < 2 && frame.pc.GetCash(frame.playerID)> 0) {

                if (e.getSource() == frame.BuyApple) {
                    String a = frame.pc.buy(frame.playerID, Stock.Apple, Integer.parseInt(frame.AppleText.getText()));
                    CashChcek(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyBP) {
                    String a =frame.pc.buy(frame.playerID, Stock.BP, Integer.parseInt(frame.BPText.getText()));
                    CashChcek(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyCisco) {
                    String a = frame.pc.buy(frame.playerID, Stock.Cisco, Integer.parseInt(frame.CiscoText.getText()));
                    CashChcek(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyDell) {
                    String a = frame.pc.buy(frame.playerID, Stock.Dell, Integer.parseInt(frame.DellText.getText()));
                    CashChcek(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyEricsson) {
                    String a = frame.pc.buy(frame.playerID, Stock.Ericsson, Integer.parseInt(frame.EricssonText.getText()));
                    CashChcek(a);
                    frame.TradeCounter++;
                }


            } else if (frame.TradeCounter == 2) {

                //frame.TradeCounter = 0;
                frame.PhaseCounter = 2;
                frame.Update();
                frame.setpanel(frame.TradePanel);
            }
        } catch (IllegalArgumentException er){
            JOptionPane.showMessageDialog(frame,
                    "Check the Amount to Buy ",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE);
        }
        if (e.getSource() == frame.Skip) {
            frame.PhaseCounter = 2;
            frame.Update();
            frame.setpanel(frame.TradePanel);
        }
    }
    private int boolconvInt(boolean value){
        return value?1:0;
    }
    private void Selling(ActionEvent e){
        try {
            if (frame.TradeCounter < 2) {

                if (e.getSource() == frame.BuyApple) {
                    String a = frame.pc.sell(frame.playerID, Stock.Apple, Integer.parseInt(frame.AppleText.getText()));
                    StockCheck(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyBP) {
                    String a =  frame.pc.sell(frame.playerID, Stock.BP, Integer.parseInt(frame.BPText.getText()));
                    StockCheck(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyCisco) {
                    String a = frame.pc.sell(frame.playerID, Stock.Cisco, Integer.parseInt(frame.CiscoText.getText()));
                    StockCheck(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyDell) {
                    String a =  frame.pc.sell(frame.playerID, Stock.Dell, Integer.parseInt(frame.DellText.getText()));
                    StockCheck(a);
                    frame.TradeCounter++;
                } else if (e.getSource() == frame.BuyEricsson) {
                    String a =   frame.pc.sell(frame.playerID, Stock.Ericsson, Integer.parseInt(frame.EricssonText.getText()));
                    StockCheck(a);
                    frame.TradeCounter++;
                }
            } else if (frame.TradeCounter == 2) {
                frame.TradeCounter = 0;
                frame.PhaseCounter = 3;
                frame.Update();
                frame.setpanel(frame.votepanel);
            }
        }

        catch (IllegalArgumentException err){
            JOptionPane.showMessageDialog(frame,
                    "Check the Amount to Sell ",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE);
        }
        if (e.getSource() == frame.Skip) {
            frame.TradeCounter = 0;
            frame.PhaseCounter = 3;
            frame.Update();
            frame.setpanel(frame.votepanel);
        }
    }
    private void Voting(ActionEvent e){

        if(e.getSource().equals(frame.Accept)){
            int counter = 0;
            int [] counter2 = new int[5];
            boolean samevalue = false;
                counter = boolconvInt(frame.AppleCheck.isSelected())+ boolconvInt(frame.BPCheck.isSelected())+ boolconvInt(frame.CiscoCheck.isSelected())
                        +boolconvInt(frame.DellCheck.isSelected())+boolconvInt(frame.EricssonCheck.isSelected())
                        + boolconvInt(frame.AppleNoCheck.isSelected())+ boolconvInt(frame.BPNoCheck.isSelected())+ boolconvInt(frame.CiscoNoCheck.isSelected())
                        +boolconvInt(frame.DellNoCheck.isSelected())+boolconvInt(frame.EricssonNoCheck.isSelected());
            for(int i =0; i < 5; i++){
                if(i ==0){
                    counter2[i] = boolconvInt(frame.AppleCheck.isSelected())+boolconvInt(frame.AppleNoCheck.isSelected());
                }
                if(i==1){
                    counter2[i] = boolconvInt(frame.BPCheck.isSelected())+boolconvInt(frame.BPNoCheck.isSelected());
                }
                if(i == 2){
                    counter2[i] = boolconvInt(frame.CiscoCheck.isSelected())+boolconvInt(frame.CiscoNoCheck.isSelected());
                }
                if(i == 3){
                    counter2[i] = boolconvInt(frame.DellCheck.isSelected())+boolconvInt(frame.DellNoCheck.isSelected());
                }
                if(i == 4){
                    counter2[i] = boolconvInt(frame.EricssonCheck.isSelected())+boolconvInt(frame.EricssonNoCheck.isSelected());
                }
            }
            for(int i : counter2){
                if (i == 2){
                    samevalue = true;
                }
            }

            if (counter == 2 && samevalue == false) {
                frame.pc.vote(frame.playerID,Stock.Apple,true, frame.AppleCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.BP,true, frame.BPCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Cisco,true, frame.CiscoCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Dell,true ,frame.DellCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Ericsson,true, frame.EricssonCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Apple,false, frame.AppleNoCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.BP,false, frame.BPNoCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Cisco,false, frame.CiscoNoCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Dell,false ,frame.DellNoCheck.isSelected());
                frame.pc.vote(frame.playerID,Stock.Ericsson,false, frame.EricssonNoCheck.isSelected());
                System.out.println(frame.pc.Finished(frame.playerID));
                frame.PhaseCounter = 4;
                frame.Reset();
                frame.setpanel(frame.WaitingPanel);
                for(AI ai: bots){
                    ai.Update();
                }
            }
            else{
                JOptionPane.showMessageDialog(frame,
                        "Only Two Options need to be selected ",
                        "Warning",
                        JOptionPane.WARNING_MESSAGE);
            }
        }
        else if(e.getSource().equals(frame.SkipVote)){
            System.out.println(frame.pc.Finished(frame.playerID));
            frame.PhaseCounter = 4;
            frame.Reset();
            frame.setpanel(frame.WaitingPanel);
            for(AI ai: bots){
                ai.Update();
            }
        }

    }
    private void Checking(ActionEvent e){

        if(e.getSource() == frame.wait){
            for(AI ai: bots){
                ai.Update();

            }
            if(frame.RoundNumber != frame.pc.GetRound()){


                    frame.RoundNumber = frame.pc.GetRound();
                    frame.Round.setText("Round:: " + frame.RoundNumber);
                    frame.Update();
                frame.PhaseCounter = 1;
                frame.setpanel(frame.TradePanel);

            }
            else {
                if (frame.RoundNumber == 5) {
                    frame.results.setText(frame.pc.Getresult());
                    frame.Stats.removeAll();
                    frame.setpanel(frame.ResultsPanel);
                }
            }

        }


    }
}
