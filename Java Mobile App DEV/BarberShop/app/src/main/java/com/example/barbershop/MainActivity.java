package com.example.barbershop;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Build;
import android.os.Handler;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.AppCompatButton;
import android.text.method.HideReturnsTransformationMethod;
import android.text.method.PasswordTransformationMethod;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.FirebaseApp;


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "LoginActivity";
    private static final int REQUEST_SIGNUP = 0;
    Controller controller;
    Button button;
    EditText email;
    EditText pass;
    CheckBox checkBox;
    ProgressDialog pd;
    ImageView logo;
    private CheckBox passcheck;

    public void setupprogress(User user
    ) {
        final User us = user;
        pd.setMessage("Processing");
        pd.show();
        BackGround.firebaseController.FetchUser(us);
        Runnable runnable = new Runnable() {
            @Override
            public void run() {


                if(BackGround.firebaseController.results != null){
                    BackGround.firebaseController.UpdateUser(BackGround.firebaseController.results,BackGround.firebaseController.results.getDocuments().get(0).getReference().getParent().getPath());
                    BackGround.CurrentUser = BackGround.firebaseController.returnUser();
                //if (controller.checkWithentries(us, getBaseContext())) {
                    button.setEnabled(true);
                    setResult(RESULT_OK);
                    Toast.makeText(getBaseContext(), "you are logged in", Toast.LENGTH_LONG).show();
                    Intent intent = new Intent(getApplicationContext(), main_menu.class);
                    startActivityForResult(intent, REQUEST_SIGNUP);
                    finish();
                } else {

                    Toast.makeText(getBaseContext(), "Could not log you in :(", Toast.LENGTH_LONG).show();

                    button.setEnabled(true);
                }

                pd.dismiss();
            }
        };

        new Handler().postDelayed(runnable, 4000);


    }
    private void changetoMainMenu(){
        Intent intent = new Intent(getApplicationContext(), MainMenu.class);
        startActivityForResult(intent, 1);
    }

    @Override
    protected void onPause() {

        super.onPause();
    }
    @Override
    protected void onResume(){
        super.onResume();

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if(BackGround.CurrentUser != null){
            changetoMainMenu();
        }
        BackGround.firebaseController = new FirebaseController(this);


        //changetocalender();
        setContentView(R.layout.activity_main);
        TextView textView = this.findViewById(R.id.signuplink);
        controller = new Controller();
        button = this.findViewById(R.id.loginbut);
        System.out.println(button);
        email = this.findViewById(R.id.login_email);
        pass = this.findViewById(R.id.login_pass);
        passcheck = this.findViewById(R.id.passwordcheck);
        pass.setTransformationMethod(PasswordTransformationMethod.getInstance());

        setCoord();
        pd = new ProgressDialog(MainActivity.this);
        checkBox = this.findViewById(R.id.login_checkbox);
        logo = this.findViewById(R.id.logo);
        Display display = getWindowManager(). getDefaultDisplay();
        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);
        int h = dm.heightPixels;
        int w = dm.widthPixels;
        logo.setMaxWidth(w-24);
        logo.getLayoutParams().height = h/3;
        logo.setMaxHeight(h/4);
        textView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(getApplicationContext(), SignupActivity.class);
                startActivityForResult(intent, REQUEST_SIGNUP);
            }
        });

        button.setOnClickListener(new View.OnClickListener() {
                                   @Override
                                   public void onClick(View v) {
                                       User.UserType ut;
                                       if (checkBox.isChecked()) {

                                           ut = User.UserType.Barber;
                                       } else {
                                           ut = User.UserType.Customer;
                                       }
                                       User user = new User(email.getText().toString(), pass.getText().toString(), "", ut);
                                       setupprogress(user);
                                   }


                               }
        );


    }


    public void setCoord(){

        passcheck.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {

            @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN)
            @Override
            public void onGlobalLayout() {

                // Removing layout listener to avoid multiple calls
                if(Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                    passcheck.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                }
                else {
                    passcheck.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                }

                passcheck.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
                    @Override
                    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                        if(isChecked){

                            pass.setTransformationMethod(HideReturnsTransformationMethod.getInstance());
                        }
                        else{
                            pass.setTransformationMethod(PasswordTransformationMethod.getInstance());


                        }
                    }
                });
            }
        });
    }
}