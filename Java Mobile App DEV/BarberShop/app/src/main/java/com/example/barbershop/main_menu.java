package com.example.barbershop;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.print.PrintAttributes;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.common.api.Api;
import com.google.firebase.FirebaseApp;


public class main_menu extends AppCompatActivity {

    private String TAG = "MAINMENU";
    private GridLayout gl;
    private ImageView book;
    private ImageView barber;
    private ImageView settings;
    private ImageView schedule;
    private ProgressDialog pd;
    private boolean exited = false;
    private int x,y;
    private void setIntents(){
        book.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                BackGround.firebaseController.LoadAccordingToUser(BackGround.CurrentUser.getEmail());

                    pd.setMessage("Fetching bookings...");
                    pd.show();
                    Runnable runnable = new Runnable() {
                        @Override
                        public void run() {

                            if (BackGround.firebaseController.results != null) {
                                BackGround.CurrentBookings = BackGround.firebaseController.returnCustomerBookings();
                                Log.i(TAG, String.valueOf(BackGround.CurrentBookings.size()));
                                Intent intent = new Intent(getApplicationContext(), Bookings.class);
                                startActivityForResult(intent, 1);


                            }
                            else{
                                Toast.makeText(getBaseContext(), "No bookings available to inspect", Toast.LENGTH_LONG).show();
                            }
                            pd.dismiss();

                        }
                    };
                    new Handler().postDelayed(runnable, 4000);

                }

        });
        barber.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), SearchBarberShop.class);
                startActivityForResult(intent, 1);
            }
        });
        settings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), Settings.class);
                startActivityForResult(intent, 1);
            }
        });
        schedule.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                BackGround.firebaseController.timetableProcess();
                pd.setMessage("Updating...");
                pd.show();
                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        if(BackGround.tempaval != null){

                            Intent intent = new Intent(getApplicationContext(), CalenderBooking.class);
                            startActivityForResult(intent, 1);
                            finish();


                        }
                        pd.dismiss();

                    }
                };


                new Handler().postDelayed(runnable, 4000);


            }
        });

    }

    @Override
    protected void onResume(){

        super.onResume();
        DisplayMetrics dm = new DisplayMetrics();
        book.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {

            @Override
            public void onGlobalLayout() {

                // Removing layout listener to avoid multiple calls
                if(Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                    book.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                }
                else {
                    book.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                }

                x = book.getWidth();
                getWindowManager().getDefaultDisplay().getMetrics(dm);
                int width = dm.widthPixels/(dm.widthPixels/150);
                int padding = (dm.widthPixels *10/100);
                Log.i(TAG, String.valueOf(x));
                gl.setX((dm.widthPixels/2 )-(book.getX()));
                gl.setPadding(0,padding,0,0);
            }
        });



    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);


        setContentView(R.layout.activity_main_menu);
        Display display = getWindowManager(). getDefaultDisplay();
        gl = findViewById(R.id.menugrid);
        book = findViewById(R.id.bookings);
        barber = findViewById(R.id.barbershop);
        settings = findViewById(R.id.settings);
        schedule = findViewById(R.id.book);
        pd = new ProgressDialog(this);


        //gl.setX((dm.widthPixels/2)- (dm.widthPixels*30/100));
       //



        setIntents();



    }


}
