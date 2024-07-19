package com.example.barbershop;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

public class Controller {
    private User users;
    Context context;
    public Controller(){
        users = new User();
    }


    public boolean savetoFile(final User user, Context context
    ){
        this.context =  context;
        loadfromFile(context);

        if(user.getEmail() == null  || user.getName() == null || user.getPass() == null
         ||user.getEmail().length() == 0|| user.getName().length() == 0 || user.getPass().length() == 0){
            return false;
        }

        try{

            File f = new File(context.getFilesDir(),"entries");
            f.delete();
            f.createNewFile();

            FileOutputStream file = new FileOutputStream(f,false);
            ObjectOutputStream os = new ObjectOutputStream(file);
            os.writeObject(user);
            os.flush();
            os.close();
            file.close();
            loadfromFile(context);
        }catch (IOException e){
            //e.printStackTrace();
            return false;
        }

        return true;
    }
    public void loadfromFile( Context context){

        try {
            File f = new File(context.getFilesDir(),"entries");

            FileInputStream file = new FileInputStream(f);
            ObjectInputStream in = new ObjectInputStream(file);
            users = (User)in.readObject();
            file.close();
            in.close();
        }catch (Exception e){
           e.printStackTrace();
        }

    }

    public void setUsers(User users) {
        this.users = users;
    }

    public User getUsers() {
        return users;
    }
}
