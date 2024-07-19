package com.example.barbershop;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

public class SignupActivity extends AppCompatActivity {
    private static final int REQUEST = 0;
    private Controller controller;
    private TextView textView;
    private Button btn ;
    private EditText name;
    private EditText email;
    private String TAG = "SignUP";
    private EditText pass ;
    private ProgressDialog pd;
    private ImageView logo;
    private CheckBox checkBox;
    public void setupprogress(User user
    ){
        final User us = user;
        pd.setMessage("Processing");
        pd.show();
         final boolean created = BackGround.firebaseController.addnewUser(us);
        Runnable runnable = new Runnable() {
            @Override
            public void run() {

                if(created){
                //if(controller.savetoFile(us,getBaseContext())){
                    btn.setEnabled(true);
                    setResult(RESULT_OK);
                    finish();
                }
                else{

                    Toast.makeText(getBaseContext(),"Could not sign up, Email already Registered", Toast.LENGTH_LONG).show();

                    btn.setEnabled(true);
                }

                pd.dismiss();
            }
        };

        new Handler().postDelayed(runnable,4000);


    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        controller = new Controller();
        setContentView(R.layout.signupactivity);
         textView = this.findViewById(R.id.link_login);
         btn = this.findViewById(R.id.btn_signup);
         name = this.findViewById(R.id.name);
         email = this.findViewById(R.id.email);
         pass = this.findViewById(R.id.password);
        checkBox = this.findViewById(R.id.signup_checkbox);
         pd = new ProgressDialog(SignupActivity.this);
        Display display = getWindowManager(). getDefaultDisplay();
        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);
        logo = this.findViewById(R.id.logo);
        int h = dm.heightPixels;
        int w = dm.widthPixels;
        logo.setMaxWidth(w-24);
        logo.getLayoutParams().height = h/3;
        logo.getLayoutParams().width = w;
        logo.setMaxHeight(h/4);

        textView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                startActivityForResult(intent, REQUEST);
            }
        });
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                User.UserType userType;
                if(checkBox.isChecked()){
                    userType = User.UserType.Barber;
                }
                else{
                    userType = User.UserType.Customer;
                }
                User user = new User(email.getText().toString(), pass.getText().toString(), name.getText().toString(), userType);

                setupprogress(user);
            }


        });

    }
}


