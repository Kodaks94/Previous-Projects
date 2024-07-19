package com.example.barbershop;

import android.content.Context;
import android.os.Build;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.util.Log;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.FirebaseApp;
import com.google.firebase.Timestamp;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FieldValue;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.FirebaseFirestoreSettings;
import com.google.firebase.firestore.Query;
import com.google.firebase.firestore.QueryDocumentSnapshot;
import com.google.firebase.firestore.QuerySnapshot;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;


public class FirebaseController {

    String TAG = "DATABASE";
    private FirebaseFirestore database;
    public static   QuerySnapshot results;
    public  User CurrentUser;

    public FirebaseController(Context context){
        FirebaseApp.initializeApp(context);
        database = FirebaseFirestore.getInstance();
        FirebaseFirestoreSettings settings = new FirebaseFirestoreSettings.Builder()
                .setTimestampsInSnapshotsEnabled(true)
                .build();
        database.setFirestoreSettings(settings);

    }
    public FirebaseController(){

    }
    public String UserTypeConverter(User.UserType userType){
        switch (userType) {

            case Customer:
                 return "Customers";

            case Barber:
                return "Barbers";
            default:
                return null;
        }


    }
    public void FetchUser(User user) {

        results = null;
        CurrentUser = null;
        String usertype = UserTypeConverter(user.getUserType());
        Query query = database.collection(usertype).whereEqualTo("Email", user.getEmail()).whereEqualTo("Password", user.getPass());
        query.get().addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onComplete(@NonNull Task<QuerySnapshot> task) {
                if (Objects.requireNonNull(task.getResult()).size() > 0) {
                    results = task.getResult();


                }
            }

        });


    }

    public void timetableProcess(){

        results = null;

        BackGround.tempaval = null;

        Query query = database.collection("Barbers").orderBy("Email");
        TailoredOnCompleteListener listener = new TailoredOnCompleteListener();
        query.get().addOnCompleteListener(listener);

        Runnable runnable = new Runnable() {
            @Override
            public void run() {

                if (results != null) {
                    BackGround.tempaval = new ArrayList<>();

                    for (DocumentSnapshot a : results.getDocuments()) {
                        try {

                            for (Timestamp timestamp : (ArrayList<Timestamp>) a.get("AvailableTimes")) {

                                BookTime bookTime = new BookTime((String) a.get("Email"), timestamp);
                                BackGround.tempaval.add(bookTime);

                            }

                        } catch (Exception e) {

                        }

                    }

                }
            }
            };
        new Handler().postDelayed(runnable, 4000);
        }






    public User returnUser(){

        return CurrentUser;
    }
    public void UpdateUser(QuerySnapshot qs,String ut){


        User.UserType userType = null;
        if(ut.equals("Customers")){
            userType = User.UserType.Customer;
        }
        else if(ut.equals("Barbers")){
            userType = User.UserType.Barber;
        }
         DocumentSnapshot user = qs.getDocuments().get(0);

        CurrentUser = new User( user.getId() ,user.get("Email").toString(), user.get("Password").toString(),user.get("Name").toString() , userType);


    }
    public void UpdateAfterChange(){

        FetchUser(CurrentUser);


    }

    public void setAvailableTimestamp(Timestamp entrydate){

        DocumentReference documentReference = database.collection("Barbers").document(BackGround.CurrentUser.getID());

        documentReference.update("AvailableTimes", FieldValue.arrayUnion(entrydate));


    }
    public List<BookTime> returnCustomerBookings(){
        List<BookTime> booktimes = new ArrayList<>();
        for(QueryDocumentSnapshot q: results){

            BookTime temp = new BookTime(
                    q.getString("BarberEmail"),
                    q.getString("CustomerEmail"),
                    q.getTimestamp("BookTime")

            );
            temp.setID(q.getId());
            booktimes.add(temp);

        }
        return booktimes;
    }

    public void LoadAccordingToUser(String user){

        String useremail;
        if(BackGround.CurrentUser.getUserType() == User.UserType.Customer){
            useremail = "CustomerEmail";
        }
        else{
            useremail = "BarberEmail";
        }
        Query query = database.collection("BookingTimes").whereEqualTo(useremail, user);
        results = null;
        query.get().addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onComplete(@NonNull Task<QuerySnapshot> task) {
                if (Objects.requireNonNull(task.getResult()).size() > 0) {

                    results = task.getResult();
                }
            }

        });

    }





    public boolean DoesUserExist(User user){

        boolean result = false;
        results = null;

        String usertype =UserTypeConverter(user.getUserType());


        Query query = database.collection(usertype).whereEqualTo("Email", user.getEmail());
        query.get().addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onComplete(@NonNull Task<QuerySnapshot> task) {
                if(Objects.requireNonNull(task.getResult()).size() > 0){
                    results = task.getResult();
                }
            }
        });
        if( results == null){
            result  = false;

        }
        else {
            result = true;
        }

        return result;
    }


    public boolean addnewUser(User user) {


        if (!DoesUserExist(user)) {


            Map<String, Object> Details = new HashMap<>();

            Details.put("Name", user.getName());
            Details.put("Email", user.getEmail());
            Details.put("Password", user.getPass());
            Details.put("Bookings", user.getTime());
            final Task<Void> NUser = database.collection(UserTypeConverter(user.getUserType())).document().set(Details);
            NUser.addOnSuccessListener(new OnSuccessListener<Void>() {
                @Override
                public void onSuccess(Void aVoid) {

                }
            }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    Log.i(TAG, "FAILED TO MAKE NEW USER");
                }
            });

            return true;
        }

        return false;
    }

    public void searchforBarber( String name){

        results = null;
        Query query = database.collection("Barbers").whereEqualTo("Name", name);
        query.get().addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onComplete(@NonNull Task<QuerySnapshot> task) {
                if(Objects.requireNonNull(task.getResult()).size() > 0){
                    results = task.getResult();
                }
            }
        });
    }



    public void AddBookingTime(BookTime time){


        Map<String, Object> Details = new HashMap<>();

        Details.put("BarberEmail", time.getBarber());
        Details.put("CustomerEmail", time.getCustomer());
        Details.put("BookTime", time.getTimestamp());
        final Task<Void> NUser = database.collection("BookingTimes").document().set(Details);
        NUser.addOnSuccessListener(new OnSuccessListener<Void>() {
            @Override
            public void onSuccess(Void aVoid) {


            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Log.i(TAG, "Failed to enter the booktime");
            }
        });
        results = null;
        Query query = database.collection("Barbers").whereEqualTo("Email", time.getBarber());

        query.get().addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onComplete(@NonNull Task<QuerySnapshot> task) {
                if(Objects.requireNonNull(task.getResult()).size() > 0){

                    results = task.getResult();
                }
            }
        });



    }
    public void updateTheReferences(BookTime time){

        Log.i(TAG,String.valueOf(results.getDocuments().get(0).get("Email")));
        DocumentReference documentReference = database.collection("Barbers").document(results.getDocuments().get(0).getId());

        documentReference.update("AvailableTimes", FieldValue.arrayRemove(time.getTimestamp()));
    }


    public void DeleteAbooking(String id){

        database.collection("BookingTimes").document(id).delete().addOnSuccessListener(new OnSuccessListener<Void>() {
            @Override
            public void onSuccess(Void aVoid) {

            }
        });


    }
    public void DeleteUser(String id, User.UserType userType){

        String user = UserTypeConverter(userType);

        database.collection(user).document(id).delete().addOnSuccessListener(new OnSuccessListener<Void>() {
            @Override
            public void onSuccess(Void aVoid) {

            }
        });
    }

}


