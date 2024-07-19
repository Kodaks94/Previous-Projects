using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VR_Mouse_replacement : MonoBehaviour
{

    [SerializeField] private string tag = "Stimuli_obj";
    // Start is called before the first frame update

    private Transform t;
    
    void Start()
    {
      

    }

    // Update is called once per frame
    void Update()
    {


        if (t != null)
        {
            t.GetComponent<Hover_UI>().notSelected();
            t = null;

        }
        var ray = Camera.main.ScreenPointToRay(this.transform.position);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            var s = hit.transform;
            if (s.CompareTag(tag))
            {
                var sr = s.GetComponent<Renderer>();
                if (sr != null)
                {
                    s.GetComponent<Hover_UI>().Selected();

                }
                t = s;
            }
        }
        
    }
}
