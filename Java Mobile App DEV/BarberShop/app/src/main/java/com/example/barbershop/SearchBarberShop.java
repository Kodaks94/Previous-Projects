package com.example.barbershop;

import android.app.ProgressDialog;
import android.graphics.Color;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.QuerySnapshot;

import java.util.ArrayList;
import java.util.List;

public class SearchBarberShop extends AppCompatActivity {
    private String TAG = "SEARCH";
    private Button SearchButton;
    private EditText editText;
    private List<String> barbers;
    private ProgressDialog pd;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search_barber_shop);
        SearchButton = findViewById(R.id.search_button);
        editText = findViewById(R.id.textsearch);
        barbers = new ArrayList<>();
        pd = new ProgressDialog(this);
        pd.setMessage("Searching...");

        SearchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Search(editText.getText().toString());

            }
        });


    }

    private void Search(String name){
        BackGround.firebaseController.searchforBarber(name);
        pd.show();
    Runnable runnable = new Runnable() {
        @Override
        public void run() {

            if(BackGround.firebaseController.results != null){

                QuerySnapshot qs = BackGround.firebaseController.results;
                if(qs.getDocuments().size() == 0){
                    Toast.makeText(getBaseContext(), "User does not exist", Toast.LENGTH_LONG).show();
                    return;
                }
                int i = 1;
                LinearLayout linearLayout = (LinearLayout)findViewById(R.id.searchlayout);
                LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT);
                linearLayout.removeAllViews();

                for(DocumentSnapshot ds : qs.getDocuments()){

                    Button textView = new Button(SearchBarberShop.this);
                    textView.setId(i);
                    textView.setLayoutParams(params);
                    textView.setText(i + "-- " +"Email: "+ ds.get("Email")+ "|| Name :" + ds.get("Name"));
                    i++;
                    textView.setBackgroundColor(Color.parseColor("#BD9265"));
                    textView.setTextColor(Color.BLACK);
                    linearLayout.addView(textView, params);

                }
            }

            pd.dismiss();

        }
    };
        new Handler().postDelayed(runnable, 4000);



    }
}
