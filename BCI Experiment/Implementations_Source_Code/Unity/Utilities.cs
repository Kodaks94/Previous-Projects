using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Utilities : MonoBehaviour
{
    int frameCount = 0;
    float dt, fps = 0;
    float rate = 4;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        frameCount++;
        dt += Time.deltaTime;
        if(dt > 1.0 / rate)
        {
            fps = frameCount / dt;
            frameCount = 0;
            dt -=  1 / rate;

        }
        
    }
    public float get_fps()
    {
        return fps;
    }
    public float get_frameCount()
    {
        return frameCount;
    }
}
