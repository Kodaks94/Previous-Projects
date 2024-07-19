package com.example.barbershop;

import android.os.Build;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.util.Log;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.firestore.Query;
import com.google.firebase.firestore.QuerySnapshot;

import java.util.Objects;

public class TailoredOnCompleteListener implements OnCompleteListener<QuerySnapshot> {
    public  boolean finished = false;
    String TAG = "LISTENER";
    public TailoredOnCompleteListener(){

    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    public void onComplete(@NonNull Task<QuerySnapshot> task) {

        if (Objects.requireNonNull(task.getResult()).size() > 0) {

            FirebaseController.results = task.getResult();

        }
        finished = true;

    }


}
