using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseOver : MonoBehaviour {

	// Use this for initialization

	public GameObject light;
	void Start () {

		if (light == null)
		{
			light = gameObject.transform.GetChild(0).gameObject;

		



		}

	}
	
	// Update is called once per frame
	void Update () {
		
	}

	private void OnMouseUp()
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
}
