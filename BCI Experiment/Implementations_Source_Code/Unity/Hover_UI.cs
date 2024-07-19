using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;

public enum Direction { Up, down, left, right}
public enum LightAction { Red,Blue,Green,OFF_ON, Yellow, Magneta, Cyan , Unknown }

[System.Serializable]
public class buttonsconfig
{

    public Direction direction;
    public LightAction action;
    public float HZ;
    public string text = "";
    [System.NonSerialized]
    public float dt = 0;
    [System.NonSerialized]
    public float frame = 0;
    [System.NonSerialized]
    public Button stimuli;
    public double change_time = 0;

}

public class Hover_UI : MonoBehaviour
{
    private int LS_frame = 0, CS_frame = 0;
    private float next_sec;
    private Stim_code stim;
    private Tcp_manager tcp;
    public buttonsconfig[] buttons;
    public bool Java_integrated;
    public string ObjString;
    public Text ObjText;
    public float f_time;
    public bool Is_shown;
    Utilities Util;
    Experiment experiment;
    public bool is_experimenting;
    private Vector3 Original_pos;
    private float distance = 2f;
    float speed = 2f;
    float error = 0.1f;
    private bool flash = false;
    private Process foo;
    public Button ButtonPrefab;
    public GameObject light;

    //shoulnt be multiple 
    int[] frame_delay = { 10, 26 };
    private bool running = false;

    // TODO : actual value of the frequency must be checked. 10 seconds,  BUILD TOO
    // For more than 2 flashings: 
    // one way is to use a separate screen
    // 3 hrtz and ghost images with multiples....
    // Start is called before the first frame update
    void Start()
    {
        next_sec = (Time.time * 1000) + 1000;
        tcp = GameObject.Find("GameScripts").GetComponent<Tcp_manager>();

        if (light == null)
        {
            light = gameObject.transform.GetChild(0).gameObject;





        }
        Util = GameObject.FindGameObjectWithTag("Util").GetComponent<Utilities>();
        experiment = GameObject.FindGameObjectWithTag("Util").GetComponent<Experiment>();
        experiment.next_exp = true;
        ObjText = GameObject.Find("Text").GetComponent<Text>();
        ObjText.color = Color.clear;

        CanvasScript c = gameObject.GetComponentInChildren<CanvasScript>() ;

        Original_pos = c.transform.position;
        foo = new Process();
        foo.StartInfo.FileName = "C:/Users/Mahrad/Desktop/CE901/ce901_pisheh_var_m/Flasher/out/artifacts/Flasher_jar/Flasher.jar";
        foo.StartInfo.Arguments = "2 15";
        foo.StartInfo.WindowStyle = ProcessWindowStyle.Normal;

        foreach (buttonsconfig b in buttons)
        {


            b.stimuli = c.Create_buttons(ButtonPrefab.gameObject, b).GetComponent<Button>();


        }


    }


    public void increase_dt()
    {
        foreach (buttonsconfig s in buttons)
        {
            s.dt += Time.deltaTime;

        }

    }


    private bool Stim_checker(int i)
    {
        if (i <= buttons.Length)
        {
            return true;

        }
        else return false;


    }
  
    private void if_exp_increase_ST(Stim_code stim)
    {
        
        if (is_experimenting)
        {
            if (currentrec != null)
            {
                currentrec.increase_tries();
                if (currentrec.stim_Code.Equals(stim))
                {
                    currentrec.increase_ST();
                    currentrec.achieved = true;


                }
            }
        }
            

    }
    private LightAction action_from_tcp(Stim_code s)
    {
        LightAction action = LightAction.Unknown;
        switch (s)
        {
            case Stim_code.Unknown:
                break;

            case Stim_code.SL1:
                
                if_exp_increase_ST(s);
                if (Stim_checker(1)) action = buttons[0].action;
                break;
            case Stim_code.SL2:
            
                if_exp_increase_ST(s);
                if (Stim_checker(2)) action = buttons[1].action;
                break;
            case Stim_code.SL4:
              
                if_exp_increase_ST(s);
                if (Stim_checker(3)) action = buttons[2].action;
                break;





            case Stim_code.SL3:
                action = LightAction.Unknown;
                break;
           
            case Stim_code.SL5:
                action = LightAction.Unknown;

                break;
            case Stim_code.SL6:
                action = LightAction.Unknown;

                break;
            case Stim_code.SL7:
                action = LightAction.Unknown;

                break;
            default:
                action = LightAction.Unknown;
                break;
          
        }

        return action;
    }



    public void  turn_light()
    {
        if (light.activeSelf)
        {
            light.SetActive(false);
        }
        else
        {
            light.SetActive(true);
        }
    }
    public void lit_action(LightAction action)
    {

        switch (action)
        {
            case LightAction.Red:
                light.GetComponent<Light>().color = Color.red;
            
                break;
            case LightAction.Blue:
                light.GetComponent<Light>().color = Color.blue;
                
                break;
            case LightAction.Green:
                light.GetComponent<Light>().color = Color.green;
               
                break;
            case LightAction.OFF_ON:
                turn_light();
                break;
            case LightAction.Yellow:
                light.GetComponent<Light>().color = Color.yellow;
              
                break;
            case LightAction.Magneta:
                light.GetComponent<Light>().color = Color.magenta;
                
                break;
            case LightAction.Cyan:
                light.GetComponent<Light>().color = Color.cyan;
               
                break;
            case LightAction.Unknown:
                break;
        }
    }


    Records currentrec= null;
    public void if_flash()
    {

        if (flash)
        {
            
            if (is_experimenting )
            {
                if (experiment.next_exp)
                {
                    currentrec = experiment.Start_experiment(currentrec);
                    if (currentrec == null) { is_experimenting = false; return; }
                    experiment.next_exp = false;
                }
               
                else{

                    if (currentrec.Timer > 50 || currentrec.achieved) { experiment.next_exp = true; currentrec.achieved = false; }
                }
                currentrec.TimerUpdate();
            }

          
            
            stim = tcp.mymessage;

            lit_action(action_from_tcp(stim));
            

            foreach (buttonsconfig s in buttons)
            {
                s.frame += Time.deltaTime;

                // button_flash_fixed(s); //Frame Base flickering
                button_flash_based_on_time(s); //Time based flickering

            }



        }


    }
    // Update is called once per frame
    void Update()
    {
        
        Fade_Func();
        increase_dt();
        if_flash();
        
    }

    
    private void button_flash_based_on_time(buttonsconfig b)
    {

        float when_to_flash = b.HZ;
        when_to_flash = 1 / (when_to_flash * Time.deltaTime);
       
        float current_T = Time.realtimeSinceStartup;
       
        if (current_T >= b.change_time) 
        {
           
            b.change_time += (current_T + when_to_flash);
            b.stimuli.enabled = !b.stimuli.enabled;
        }

       
    }
    private void button_flash_fixed(buttonsconfig b)
    {

       
        if(b.frame >= 1/b.HZ)
        {
           
            b.stimuli.enabled = !b.stimuli.enabled;
            b.frame = 0; 

        }
        
        


    }
  
  
    private void button_flash(buttonsconfig b)
    {
     
        if(b.dt >= b.HZ / Util.get_fps())
        {
          
            b.stimuli.enabled = !b.stimuli.enabled;
            b.dt = 0;
            
        }
       
        

    }
    public void Selected()
    {
        Is_shown = true;
        
    }
    public void notSelected()
    {
        Is_shown = false;
    }
    private void OnMouseOver()
    {

        
            Is_shown = true;
        
        
       
    }

    private void OnMouseUp()
    {
        if (Java_integrated)
        {

            foo.Start();

        }

    }

        private void OnMouseExit()
    {

       
            Is_shown = false;
      

    }


    private void buttons_controller(buttonsconfig b)
    {

     


            switch (b.direction)
            {
                case Direction.Up:
                    b.stimuli.transform.position += b.stimuli.transform.up * Time.deltaTime * speed;
                    break;
                case Direction.down:
                    b.stimuli.transform.position -= b.stimuli.transform.up * Time.deltaTime * speed;
                    break;
                case Direction.left:
                    b.stimuli.transform.position -= b.stimuli.transform.right * Time.deltaTime * speed;
                    break;
                case Direction.right:
                    b.stimuli.transform.position += b.stimuli.transform.right * Time.deltaTime * speed;
                    break;
            }


        
    }

    private void buttons_change_color()
    {

        if (Java_integrated)
        {
            
            dontflash();
        }
        else
        {

            if (Is_shown)
            {
                ObjText.text = ObjString;
                ObjText.color = Color.white;
                foreach (buttonsconfig b in buttons)
                {
                    b.stimuli.enabled = true;
                    b.stimuli.GetComponentInChildren<Text>().color = Color.Lerp(ObjText.color, Color.white, f_time * Time.deltaTime);

                    if (Vector3.Distance(Original_pos, b.stimuli.transform.position) < distance - error)
                    {
                        buttons_controller(b);
                    }
                    else
                    {
                        flash = true;
                    }
                }


            }
            else
            {

                dontflash();


            }
        }
    }

    private void  dontflash()
    {
        foreach (buttonsconfig b in buttons)
        {
            b.stimuli.enabled = false;
            b.stimuli.GetComponent<Image>().CrossFadeAlpha(0, f_time * Time.deltaTime, false);
            b.stimuli.GetComponentInChildren<Text>().color = Color.Lerp(ObjText.color, Color.clear, f_time * Time.deltaTime);
            if (Vector3.Distance(Original_pos, b.stimuli.transform.position) > 0.5f)
            {

                move(b.stimuli, 0.5f);
            }



            if (Vector3.Distance(Original_pos, b.stimuli.transform.position) > 0.5f)
            {

                move(b.stimuli, 0.5f);
            }
        }
        ObjText.color = Color.Lerp(ObjText.color, Color.clear, f_time * Time.deltaTime);
        flash = false;
    }
        private void Fade_Func()
    {


        buttons_change_color();
    }


    private void move(Button targetButton, float duration)
    {

        float elapsed = 0;
        Vector3 start_pos = targetButton.transform.position;

        while (elapsed < duration)
        {
            targetButton.transform.position = Vector3.Lerp(start_pos, Original_pos, elapsed);
            elapsed += Time.deltaTime;

          
         
        }

        targetButton.transform.position = Original_pos;




    }





    }
