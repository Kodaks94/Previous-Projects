package com.example.barbershop;

import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Bookings extends AppCompatActivity {

    String TAG = "BOOKINGS";
    Map<BookTime, Button> buttons;

    @SuppressLint({"ResourceAsColor", "WrongConstant"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bookings);
        LinearLayout linearLayout = (LinearLayout)findViewById(R.id.buttonlayout);
        int i = 1;
        buttons = new HashMap<>();
        for(BookTime t : BackGround.CurrentBookings) {
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT);

            Button button = new Button(this);
            button.setId(i);
            button.setLayoutParams(params);
            button.setText(i + "-- " + t.getBarber() + " Booked At: " + t.getTimestamp().toDate());
            i++;
            button.setBackgroundColor(Color.parseColor("#BD9265"));
            button.setTextColor(Color.BLACK);
            linearLayout.addView(button, params);
            buttons.put(t, button);
        }

        for(Map.Entry<BookTime, Button> button : buttons.entrySet()) {

            button.getValue().setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    AlertDialog.Builder builder = new AlertDialog.Builder(Bookings.this);
                    builder.setTitle("Do you want to delete this appointment?");
                    builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {

                            BackGround.firebaseController.DeleteAbooking(button.getKey().getID());
                            buttons.remove(button);
                            BackGround.CurrentBookings.remove(button.getKey());
                            Intent intent = getIntent();
                            finish();
                            startActivity(intent);
                        }


                        });

                    builder.setNegativeButton(android.R.string.no, null);
                    builder.show();

                }
            });



        }




        }


    }

