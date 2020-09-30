using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class UserDrive : MonoBehaviour
{
    // Wheel transforms
    public Transform Wheel_FL;
    public Transform Wheel_FR;
    public Transform Wheel_BL;
    public Transform Wheel_BR;

    // Constants
    private const float ForwardSpeed = 8;
    private const float TurnRadians = 0.05f;

    // Variables for Motion Model
    // Input
    private Vector2 input = new Vector2();
    private float wheelDist;
    // Output
    private Vector2 currHeading = new Vector2();
    private float turnTheta = 0;
    private float currForwardSpeed = 0;


    // Start is called before the first frame update
    void Start()
    {
        wheelDist = (Wheel_FL.position - Wheel_BL.position).magnitude;
    }

    // Update is called once per frame
    void Update()
    {
        RealAxelBicycleModel();
        UpdateRender();
        UpdatePosition();
    }


    /***** * * * * * Helpers * * * * * *****/
    public void OnPlayerMove(InputAction.CallbackContext context)
    {
        input = context.ReadValue<Vector2>();
    }

    private void RealAxelBicycleModel()
    {
        currForwardSpeed = input.y * ForwardSpeed;
        float currTurnRadians = input.x * TurnRadians;
        turnTheta = (currForwardSpeed * Mathf.Tan(currTurnRadians)) / wheelDist;        
    }

    private void UpdatePosition()
    {
        currHeading.x = currForwardSpeed * Mathf.Sin(turnTheta);
        currHeading.y = currForwardSpeed * Mathf.Cos(turnTheta);
        transform.position = transform.position + (currHeading.x * transform.right + currHeading.y * transform.up) * Time.deltaTime;
    }

    private void UpdateRender()
    {
        transform.Rotate(0, 0, -turnTheta);
        Wheel_FL.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * -turnTheta);
        Wheel_FR.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * -turnTheta);
    }

}
