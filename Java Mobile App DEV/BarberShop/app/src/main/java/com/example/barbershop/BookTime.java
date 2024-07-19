package com.example.barbershop;


import com.google.firebase.Timestamp;

public class BookTime {

    private String Barber, Customer;
    private String ID;
    private Timestamp timestamp;
    private boolean available;


    public BookTime(String Barber, String Customer, Timestamp timestamp){

        this.Barber = Barber;
        this.Customer = Customer;
        this.timestamp = timestamp;

        available = false;
    }
    public BookTime(){

    }
    public BookTime(String Barber, Timestamp timestamp){

        this.Barber = Barber;
        this.timestamp = timestamp;
        available = true;
    }

    public String getID() {
        return ID;
    }

    public void setID(String ID) {
        this.ID = ID;
    }

    public boolean isAvailable() {
        return available;
    }

    public String getBarber() {
        return Barber;
    }

    public String getCustomer() {
        return Customer;
    }

    public Timestamp getTimestamp() {
        return timestamp;
    }

    public void setAvailable(boolean available) {
        this.available = available;
    }

    public void setBarber(String barber) {
        Barber = barber;
    }

    public void setCustomer(String customer) {
        Customer = customer;
    }

    public void setTimestamp(Timestamp timestamp) {
        this.timestamp = timestamp;
    }
    public String getTimeasString(){

        return timestamp.toString();

    }
}
