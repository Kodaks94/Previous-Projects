using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CanvasScript : MonoBehaviour
{
    public Camera camera;
   
    // Start is called before the first frame update
    void Start()
    {
        camera = GameObject.FindGameObjectWithTag("MainCamera").GetComponent<Camera>();
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.LookAt(transform.position - camera.transform.rotation * Vector3.back, camera.transform.rotation * Vector3.up);
       
    }


    public GameObject Create_buttons(GameObject b, buttonsconfig bc)
    {

        GameObject childobj = Instantiate(b,gameObject.transform) as GameObject;

        childobj.GetComponentInChildren<Text>().text = bc.text;
       

 
        childobj.transform.SetParent( this.gameObject.transform);

        return childobj;

    }
}
