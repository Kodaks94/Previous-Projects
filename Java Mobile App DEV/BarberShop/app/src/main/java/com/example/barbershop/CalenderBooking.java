package com.example.barbershop;

import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.app.TimePickerDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Build;
import android.os.Handler;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CalendarView;
import android.widget.PopupMenu;
import android.widget.TimePicker;

import com.google.firebase.FirebaseApp;
import com.google.firebase.Timestamp;

import java.sql.Time;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class CalenderBooking extends AppCompatActivity {

    private static final String TAG = "CalendarActivity";
    private CalendarView mCalendarView;
    private Button back;
    BookingTimeController bookingTimeController;
    private List<BookTime> AvailableTimes;
    private List<BookTime> BarberTimes;
    private String[] Barbers, Times;
    ProgressDialog pd;
    private Map<Timestamp, String> chosentimestamps;

    public void PostingTodatabase(){


        bookingTimeController.PostUpdate();
        pd.setMessage("Updating...");
        pd.show();
        Runnable runnable = new Runnable() {
            @Override
            public void run() {

                if(FirebaseController.results != null){

                    BackGround.firebaseController.updateTheReferences(bookingTimeController.getTimeBooked());


                }
                pd.dismiss();


            }
        };


        new Handler().postDelayed(runnable, 4000);


    }

    public void updatingprocess() {



            AvailableTimes = BackGround.tempaval;
            Log.i(TAG, String.valueOf(AvailableTimes.size()));
            mCalendarView.setOnDateChangeListener(new CalendarView.OnDateChangeListener() {
                @Override
                public void onSelectedDayChange(CalendarView CalendarView, int year, int month, int dayOfMonth) {
                    final Date entrydtate = new Date(year, month, dayOfMonth);
                    String date = year + "/" + month + "/" + dayOfMonth;

                    if (BackGround.CurrentUser.getUserType() == User.UserType.Customer) {

                        BarberTimes = BookingTimeController.FindAvailableACCDate(AvailableTimes, entrydtate);
                        Barbers = BookingTimeController.ReturnAlltheBarbers(BarberTimes);
                        Barbers = new HashSet<String>(Arrays.asList(Barbers)).toArray(new String[0]);
                        AlertDialog.Builder alertbuilder = new AlertDialog.Builder(CalenderBooking.this);
                        alertbuilder.setTitle("Choose Barber:");
                        alertbuilder.setSingleChoiceItems(Barbers, -1, new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                bookingTimeController.getTimeBooked().setBarber(Barbers[i]);
                                AlertDialog.Builder timebuilder = new AlertDialog.Builder(CalenderBooking.this);
                                timebuilder.setTitle("Choose Time:");
                                chosentimestamps = BookingTimeController.ReturnBarberTimes(Barbers[i], BarberTimes);
                                Times = new String[chosentimestamps.size()];
                                chosentimestamps.values().toArray(Times);
                                timebuilder.setSingleChoiceItems(Times, -1, new DialogInterface.OnClickListener() {

                                    @Override
                                    public void onClick(DialogInterface dialogInterface, int i) {
                                        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss.SSS");
                                        Date parsedDate = null;
                                        Timestamp timestamp = null;
                                        for (Map.Entry<Timestamp, String> entry : chosentimestamps.entrySet()) {
                                            if (entry.getValue().equals(Times[i])) {
                                                timestamp = entry.getKey();
                                            }
                                        }
                                        bookingTimeController.getTimeBooked().setTimestamp(timestamp);

                                        PostingTodatabase();

                                        dialogInterface.dismiss();
                                    }
                                });
                                dialogInterface.dismiss();
                                timebuilder.show();
                            }

                        });

                        alertbuilder.show();


                    }
                    else{
                        TimePickerDialog mTimePicker;
                        mTimePicker = new TimePickerDialog(CalenderBooking.this, new TimePickerDialog.OnTimeSetListener() {
                            @Override
                            public void onTimeSet(TimePicker view, int hourOfDay, int minute) {

                                entrydtate.setHours(hourOfDay);
                                entrydtate.setMinutes(minute);
                                Timestamp timestamp = new Timestamp(entrydtate);
                                BackGround.firebaseController.setAvailableTimestamp(timestamp);
                            }

                        },8, 60, true);
                        mTimePicker.setTitle("Book Time:");
                        mTimePicker.show();

                    }
                }

                });

        }






    @Override
    protected void onCreate( Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        bookingTimeController = new BookingTimeController();
        bookingTimeController.getTimeBooked().setCustomer(BackGround.CurrentUser.getEmail());



        pd = new ProgressDialog(this);
        setContentView(R.layout.activity_calender_booking);
        mCalendarView = findViewById(R.id.calender);
        mCalendarView.setMinDate(System.currentTimeMillis()-1000);
        updatingprocess();
        back = findViewById(R.id.back);
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), main_menu.class);
                startActivityForResult(intent, 1);
            }
        });

    }




}
