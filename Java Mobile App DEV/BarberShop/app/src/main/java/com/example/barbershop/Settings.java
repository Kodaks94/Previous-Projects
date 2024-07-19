package com.example.barbershop;

import android.content.DialogInterface;
import android.content.Intent;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.style.BackgroundColorSpan;
import android.view.View;
import android.widget.Button;

public class Settings extends AppCompatActivity {

    Button AboutButton;
    Button SignoutButton;
    Button DeleteButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        AboutButton = findViewById(R.id.AboutButton);
        SignoutButton = findViewById(R.id.SignoutButton);
        DeleteButton = findViewById(R.id.DeleteButton);
        setIntents();

    }
    public void setIntents(){

        AboutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AlertDialog.Builder builder = new AlertDialog.Builder(Settings.this);
                builder.setTitle("This APP is a barbershop appointment booking app made by Mahrad Pisheh Var");
                builder.setNegativeButton(android.R.string.no, null);
                builder.show();

            }
        });

        SignoutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AlertDialog.Builder builder = new AlertDialog.Builder(Settings.this);
                builder.setTitle("Do you want to Sign Out");

                builder.setPositiveButton("yes", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                        BackGround.ResetAllInfo();
                        Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                        startActivityForResult(intent, 1);
                    }
                });

                builder.setNegativeButton(android.R.string.no, null);
                builder.show();


            }
        });


        DeleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AlertDialog.Builder builder = new AlertDialog.Builder(Settings.this);
                builder.setTitle("Do you want to delete your account");

                builder.setPositiveButton("yes", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                        BackGround.firebaseController.DeleteUser(BackGround.CurrentUser.getID(), BackGround.CurrentUser.getUserType());
                        BackGround.ResetAllInfo();

                        Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                        startActivityForResult(intent, 1);
                    }
                });

                builder.setNegativeButton(android.R.string.no, null);
                builder.show();


            }

        });
    }



}
