using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;


public class Records
{
    public float Timer;
    public Direction button_direction;
    public Stim_code stim_Code;
    public float HZ;
    public int total_Num_tries;
    public int successful_tries;
    public bool achieved;

    public Records(Direction direction, float HZ)
    {
        Timer = 0;

        this.button_direction = direction;

        achieved = false;
        this.HZ = HZ;
        this.total_Num_tries = 0;
        check_stimCode();

    }

    void check_stimCode()
    {
        switch (button_direction)
        {

            case Direction.down:
                stim_Code = Stim_code.SL3;
                break;
            case Direction.left:
                stim_Code = Stim_code.SL1;
                break;
            case Direction.right:
                stim_Code = Stim_code.SL4;
                break;
            case Direction.Up:
                stim_Code = Stim_code.SL2;

                break;
        }
    }

        public void TimerUpdate()
    {

        Timer += Time.deltaTime;

    }
    public void increase_tries()
    {

        total_Num_tries++;
    }
    public void increase_ST()
    {

        successful_tries++;
    }
    }
public enum ClassifierType
{
       LDA = 1,
       SVM = 2,
       NN = 3
}
public enum EnvironmnetType
{
    SimpleEnvironmnet = 1,
    ComplexEnvironmnet = 2
}
public enum DisplayType
{
    VR = 1,
    Screen = 2
}

public class Experiment : MonoBehaviour
{

    private List<Records> records;
    private string filename;
    private int subj_num = 1;
    private int[] Hzs = { 20, 15, 12, 10 };
    public bool Is_debugging = false;
    public bool operation = true;
    public bool next_exp;
    public EnvironmnetType environmnetType;
    public ClassifierType classifierType;
    public DisplayType displayType;
  
    // Start is called before the first frame update
    void Start()
    {
        records = new List<Records>();
        filename = "Subject_";
        
    }
    int called = 0;
   private Records experiment_cheker( )
    {
        Records newrec = null;
     
        switch (called)
        {

            case 1:
                newrec = new Records(Direction.left, Hzs[called-1]);
                Debug.Log("Starting to record, Please look at the button on the left");
                break;

            case 2:
                newrec = new Records(Direction.Up, Hzs[called-1]);
                Debug.Log("Starting to record, Please look at the button on the top");
                break;
            case 3:
                newrec = new Records(Direction.right, Hzs[called-1]);
                Debug.Log("Starting to record, Please look at the button on the right");
                break;
           
            


        }

        if(called > 3)
        {


        }

        return newrec;


    }


    public Records Start_experiment(Records currentrec)
    {
            
        if(called > 0 && currentrec != null )
        {
            add_records(currentrec);

        }
            called++;

        if (called > 3)
        {
            Export_to_file();
            operation = false;
            
        }
        if (operation)
        {
            StartCoroutine(waitTime(3));

            Records newrec = experiment_cheker();

            return newrec;
        }
        else
        {
            return null;
        }
        

    }

    IEnumerator waitTime(float seconds)
    {
        yield return new WaitForSeconds(seconds);
    }
    void add_records(Records rec)
    {

        records.Add(rec);
      
        


    }

    void Export_to_file()
    {
        string filename2 = displayType.ToString()+"-"+environmnetType.ToString()+"-"+classifierType.ToString()+"-"+filename + subj_num+".csv";
        StreamWriter outStream = System.IO.File.CreateText(filename2);
        outStream.WriteLine(lineSeperater + "Button Direction" + fieldSeperator + "Frequency" + fieldSeperator + "Successful tries" + fieldSeperator + "Total tries" + fieldSeperator + "Duration till action performed");
        foreach (Records i in records)
        {

            write_into_csv(i, outStream);

        }
        outStream.Close();

    }


    private char lineSeperater = '\n';
    private char fieldSeperator = ',';
    void write_into_csv(Records record, StreamWriter outStream)
    {
        try
        {
            

            string filename2 = filename+ subj_num;

            if (Is_debugging)
            {
                Debug.Log("Starting to write into the file " + filename2);

            }
            outStream.WriteLine(lineSeperater + record.button_direction.ToString() + fieldSeperator + record.HZ + fieldSeperator + record.successful_tries + fieldSeperator + record.total_Num_tries + fieldSeperator + record.Timer);
            
            //TextAsset file = Resources.Load<TextAsset>(filename2);
           // Debug.Log(lineSeperater + record.button_direction.ToString() + fieldSeperator + record.HZ + fieldSeperator + record.successful_tries + fieldSeperator + record.total_Num_tries + fieldSeperator + record.Timer);

            //File.AppendAllText(getPath()  + filename2 + ".txt", lineSeperater + record.button_direction.ToString()+fieldSeperator+ record.HZ +fieldSeperator+ record.successful_tries+ fieldSeperator +record.total_Num_tries+fieldSeperator+ record.Timer);
           

#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif

        }

        catch (Exception e)
        {

            Debug.Log("could not write into file" + filename+ subj_num+ e.Message);
            


        }


    }
    private static string getPath()
    {
#if UNITY_EDITOR
        return Application.dataPath;
#elif UNITY_ANDROID
return Application.persistentDataPath;
#elif UNITY_IPHONE
return GetiPhoneDocumentsPath();
#else
return Application.dataPath;
#endif
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
