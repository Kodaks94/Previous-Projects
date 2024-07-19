package com.example.barbershop;

import com.google.firebase.Timestamp;

import java.io.Serializable;
import java.sql.Time;
import java.sql.Date;
import java.util.List;
import java.util.Map;


public class User implements Serializable {
    private String ID;
    private String email, pass, name;
    private List<Timestamp> time;
    private List<Timestamp> AvailableTimes;
    public enum UserType{
        Barber,  Customer
    }

    private UserType userType;
    public User(String email,String pass,String name,UserType userType){
        super();
        this.email = email;
        this.pass = pass;
        this.name = name;
        this.userType = userType;
    }
    public User(){

    }
    public User(String ID,String email,String pass,String name,UserType userType){
        super();
        this.email = email;
        this.pass = pass;
        this.name = name;
        this.userType = userType;
        this.ID = ID;
    }
    public User( User user){
         new User(user.getEmail(), user.getPass(), user.getName(), user.getUserType());
    }
    public UserType getUserType() {
        return userType;
    }

    public void setUserType(UserType userType) {
        this.userType = userType;
    }

    public void setTime(List<Timestamp> time) {
        this.time = time;
    }

    public String getName() {
        return name;
    }

    public String getEmail() {
        return email;
    }

    public String getPass() {
        return pass;
    }

    public List<Timestamp> getTime() {
        return time;
    }

    public List<Timestamp> getAvailableTimes() {
        return AvailableTimes;
    }

    public void setAvailableTimes(List<Timestamp> availableTimes) {
        AvailableTimes = availableTimes;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPass(String pass) {
        this.pass = pass;
    }


    public String getID() {
        return ID;
    }

}

