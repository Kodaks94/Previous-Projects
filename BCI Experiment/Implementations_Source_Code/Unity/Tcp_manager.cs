using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Net.Sockets;
using System.Text;
using System;

public enum Stim_code
{
    Unknown = 0,
    SL0  = 33024,
    SL1 = 33025,
    SL2 = 33026,
    SL3 = 33027,
    SL4 = 33028,
    SL5 = 33029,
    SL6 = 33030,
    SL7 = 33031,
}

public class Tcp_manager : MonoBehaviour
{

    public int PORT = 5678;
    private Thread CRT; // client recieved thread
    private TcpClient tcpClient;
    public Stim_code mymessage;
    public bool is_debugging = true;

    // Start is called before the first frame update
    void Start()
    {
        connect();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    private void OnDestroy()
    {
        tcpClient.Close();
        CRT.Abort();
    }


    private void connect()
    {

        try
        {

            CRT = new Thread(new ThreadStart(listening));
            CRT.IsBackground = true;
            CRT.Start();
            if (is_debugging)
            Debug.Log("Client--> Successfully connected");
        }

        catch (Exception e) {

            if (is_debugging)
                Debug.Log("Client--> Cannot connect");


        }

        
    }

    private Stim_code message_converter(int code)
    {

        Stim_code a;

        if (Enum.IsDefined(typeof(Stim_code), code))
            a = (Stim_code)code;
        else
            a = Stim_code.Unknown;

        return a;
    }
    private void RecieveBytes(Byte[] b)
    {
        try
        {
            using (NetworkStream s = tcpClient.GetStream())
            {
                int l;

                while ((l = s.Read(b, 0, b.Length)) != 0)
                {
                    Debug.Log("connected");
                    var data = new byte[l];
                    Array.Copy(b, 0, data, 0, l);
                    
                    
                    int code = BitConverter.ToInt32(data, 0);
                   
                    mymessage = message_converter(code);
                    if (is_debugging)
                        Debug.Log("Message recieved: " + mymessage);
                        
                   
                      }
            }
           
        }
        catch (Exception e)
        {
           
                Debug.Log("Client--> Cannot recieve bytes"+ e);
            

        }
       
    }
    private void listening()
    {
        try
        {
            tcpClient = new TcpClient("localhost", PORT);
          
            Byte[] b = new Byte[1024];
            if (is_debugging)
                Debug.Log("Client-->  listening to localhost at port: "+ PORT);
            while (true)
            {
                RecieveBytes(b);
                if (tcpClient.Connected)
                {
                    Debug.Log("connected");
                }


            }



        }
        catch(SocketException e)
        {
            if (is_debugging)
                Debug.Log("Socket exception occured: check :>" + e);
        }


    }
}
