package com.example.barbershop;

import android.util.Log;

import com.google.firebase.Timestamp;

import java.sql.Time;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BookingTimeController {

    private BookTime timeBooked;
    private int year,month,day, hours, minutes, seconds;

    public BookingTimeController(){

        timeBooked = new BookTime();
    }

    public BookTime getTimeBooked() {
        return timeBooked;
    }
    public void setTimeBooked(BookTime timeBooked) {
        this.timeBooked = timeBooked;
    }
    public void PostUpdate(){

        BackGround.firebaseController.AddBookingTime(timeBooked);


    }





    public static List<BookTime>  FindAvailableACCDate(List<BookTime> AllAvilableTimes, Date date){


        List<BookTime> temptimes = new ArrayList<>();
        String[] datedetail = date.toString().split(" ");

        for(BookTime t : AllAvilableTimes){
            String[] availabletimes = t.getTimestamp().toDate().toString().split(" ");


            if(datedetail[1].equals(availabletimes[1]) && datedetail[2].equals(availabletimes[2])){
                temptimes.add(t);

            }
        }

        return temptimes;
    }

    /*public public List<BookTime> FindAllAvailableTimes(){

        return BackGround.firebaseController.UpdateAvailableTimes();

    }*/

   public static String[] ReturnAlltheBarbers(List<BookTime> passedTimes){


        String []barbers = new String [passedTimes.size()];
    int i = 0;
        for(BookTime t: passedTimes){

            barbers[i] = t.getBarber();
                    i++;

        }


    return barbers;
   }

   public static Map<Timestamp, String> ReturnBarberTimes(String Barber, List<BookTime> availableTimes){

        Map<Timestamp, String> temptimes =  new HashMap<>();
        for(BookTime time : availableTimes){

            if(time.getBarber().equals(Barber)){

                temptimes.put(time.getTimestamp(),time.getTimestamp().toDate().toString());

            }
        }


    return temptimes;

   }


}


