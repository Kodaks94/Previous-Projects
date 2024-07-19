package com.example.barbershop;

import java.util.ArrayList;
import java.util.List;

public class BackGround {
    public static FirebaseController firebaseController;
    public static User CurrentUser;
    public static List<BookTime> tempaval;
    public static List<BookTime> CurrentBookings;
    public static Controller controller;


    public static void ResetAllInfo(){
        CurrentUser = null;
        tempaval= null;
        CurrentBookings = null;

    }

}
