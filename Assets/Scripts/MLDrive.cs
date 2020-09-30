using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.EventSystems;


public struct CarState
{
    public Vector2 position;
    public double turnTheta;
    public double speed;
    public Vector2 goalPosition;
};

public struct CarInput
{
    public double relativeTurnTheta;
    public double relativeSpeed;
};

public struct ObstacleState
{
    public Vector2 position;
    public double radius;
}

public struct ModelStepResult
{
    public CarState carState;
    public CarInput carInput;
    public List<ObstacleState> obstacleStates;
}

public struct Layers
{
    System.Random rand;

    // Matrices
    // 3 Hidden layers for deeper learning. Don't have much reasoning.
    public List<double[][]> Matrices;
    public List<double[][]> StdDevs;
    public List<double[]> Biases;
    public List<double[]> StdDevBiases;

    // https://stackoverflow.com/questions/218060/random-gaussian-variables
    public double NextGaussian(in double mean, in double stdDev)
    {
        double u1 = 1.0f - (double)rand.NextDouble(); //uniform(0,1] random doubles
        double u2 = 1.0f - (double)rand.NextDouble();

        double randStdNormal =
            Math.Sqrt(-2.0f * Math.Log(u1)) *
            Math.Sin(2.0f * Math.PI * u2); //random normal(0,1)

        double randNormal =
            mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)

        return randNormal;
    }

    public void Init(Int32 inputSize, Int32 hiddenSize, Int32 outputSize)
    {
        rand = new System.Random();

        Matrices = new List<double[][]>(4);
        StdDevs = new List<double[][]>(4);
        Biases = new List<double[]>(4);
        StdDevBiases = new List<double[]>(4);

        // Matrix size Initialization
        Matrices.Add(new double[hiddenSize][]);
        StdDevs.Add(new double[Matrices[0].Length][]);
        for (Int32 i = 0; i < Matrices[0].Length; i++)
        {
            Matrices[0][i] = new double[inputSize];
            StdDevs[0][i] = new double[inputSize];
        }
        Biases.Add(new double[hiddenSize]);
        StdDevBiases.Add(new double[Biases[0].Length]);


        Matrices.Add(new double[hiddenSize][]);
        StdDevs.Add(new double[Matrices[1].Length][]);
        for (Int32 i = 0; i < Matrices[1].Length; i++)
        {
            Matrices[1][i] = new double[hiddenSize];
            StdDevs[1][i] = new double[hiddenSize];
        }
        Biases.Add(new double[hiddenSize]);
        StdDevBiases.Add(new double[Biases[1].Length]);


        Matrices.Add(new double[hiddenSize][]);
        StdDevs.Add(new double[Matrices[2].Length][]);
        for (Int32 i = 0; i < Matrices[2].Length; i++)
        {
            Matrices[2][i] = new double[hiddenSize];
            StdDevs[2][i] = new double[hiddenSize];
        }
        Biases.Add(new double[hiddenSize]);
        StdDevBiases.Add(new double[Biases[2].Length]);


        Matrices.Add(new double[outputSize][]);
        StdDevs.Add(new double[Matrices[3].Length][]);
        for (Int32 i = 0; i < Matrices[3].Length; i++)
        {
            Matrices[3][i] = new double[hiddenSize];
            StdDevs[3][i] = new double[hiddenSize];
        }
        Biases.Add(new double[hiddenSize]);
        StdDevBiases.Add(new double[Biases[3].Length]);

    }

    public void Init(Layers layer, in float stdDev)
    {
        rand = new System.Random();

        Matrices = new List<double[][]>(4);
        StdDevs = new List<double[][]>(4);
        Biases = new List<double[]>(4);
        StdDevBiases = new List<double[]>(4);



        // Matrix size Initialization
        Matrices.Add(new double[layer.Matrices[0].Length][]);
        StdDevs.Add(new double[layer.StdDevs[0].Length][]);
        for (Int32 i = 0; i < Matrices[0].Length; i++)
        {
            Matrices[0][i] = new double[layer.Matrices[0][i].Length];
            StdDevs[0][i] = new double[layer.StdDevs[0][i].Length];
        }
        Biases.Add(new double[layer.Biases[0].Length]);
        StdDevBiases.Add(new double[layer.Biases[0].Length]);

        Matrices.Add(new double[layer.Matrices[1].Length][]);
        StdDevs.Add(new double[Matrices[1].Length][]);
        for (Int32 i = 0; i < Matrices[1].Length; i++)
        {
            Matrices[1][i] = new double[layer.Matrices[1][i].Length];
            StdDevs[1][i] = new double[Matrices[1][i].Length];
        }
        Biases.Add(new double[layer.Biases[1].Length]);
        StdDevBiases.Add(new double[Biases[1].Length]);

        Matrices.Add(new double[layer.Matrices[2].Length][]);
        StdDevs.Add(new double[Matrices[2].Length][]);
        for (Int32 i = 0; i < Matrices[2].Length; i++)
        {
            Matrices[2][i] = new double[layer.Matrices[2][i].Length];
            StdDevs[2][i] = new double[Matrices[2][i].Length];
        }
        Biases.Add(new double[layer.Biases[2].Length]);
        StdDevBiases.Add(new double[Biases[2].Length]);

        Matrices.Add(new double[layer.Matrices[3].Length][]);
        StdDevs.Add(new double[Matrices[3].Length][]);
        for (Int32 i = 0; i < Matrices[3].Length; i++)
        {
            Matrices[3][i] = new double[layer.Matrices[3][i].Length];
            StdDevs[3][i] = new double[Matrices[3][i].Length];
        }
        Biases.Add(new double[layer.Biases[3].Length]);
        StdDevBiases.Add(new double[Biases[3].Length]);



        // Copy of values
        for (Int32 i = 0; i < Matrices.Count; i++)
        {
            for (Int32 j = 0; j < layer.Matrices[i].Length; j++)
            {
                for (Int32 k = 0; k < layer.Matrices[i][j].Length; k++)
                {
                    Matrices[i][j][k] = layer.Matrices[i][j][k] + layer.StdDevs[i][j][k] * NextGaussian(0, stdDev * stdDev);
                    StdDevs[i][j][k] = layer.StdDevs[i][j][k];
                }
            }
        }

        for (Int32 i = 0; i < Biases.Count; i++)
        {
            for (Int32 j = 0; j < layer.Biases[i].Length; j++)
            {
                Biases[i][j] = layer.Biases[i][j] + layer.StdDevBiases[i][j] * NextGaussian(0, stdDev * stdDev);
                StdDevBiases[i][j] = layer.StdDevBiases[i][j];
            }
        }
    }

    public void SetMean(in double value)
    {
        for (Int32 i = 0; i < 4; i++)
        {

            for (Int32 j = 0; j < Matrices[i].Length; j++)
            {
                for (Int32 k = 0; k < Matrices[i][j].Length; k++)
                {
                    Matrices[i][j][k] = value;
                }
            }

            for (Int32 j = 0; j < Biases[i].Length; j++)
            {
                Biases[i][j] = value;
            }

        }
    }

    public void SetStdDev(in double value)
    {
        for (Int32 i = 0; i < 4; i++)
        {

            for (Int32 j = 0; j < StdDevs[i].Length; j++)
            {
                for (Int32 k = 0; k < StdDevs[i][j].Length; k++)
                {
                    StdDevs[i][j][k] = value;
                }
            }

            for (Int32 j = 0; j < StdDevBiases[i].Length; j++)
            {
                    StdDevBiases[i][j] = value;
            }

        }
    }

    public void AddNoise(in double value)
    {
        for (Int32 i = 0; i < StdDevs.Count; i++)
        {
            for (Int32 j = 0; j < StdDevs[i].Length; j++)
            {
                for (Int32 k = 0; k < StdDevs[i][j].Length; k++)
                {
                    StdDevs[i][j][k] += value;
                }
            }
        }

        for (Int32 i = 0; i < StdDevBiases.Count; i++)
        {
            for (Int32 j = 0; j < StdDevBiases[i].Length; j++)
            {
                StdDevBiases[i][j] += value;
            }
        }
    }

}

public class MLDrive : MonoBehaviour
{
    // Goal
    public Transform Goal;

    // Wheel transforms
    public Transform Wheel_FL;
    public Transform Wheel_FR;
    public Transform Wheel_BL;
    public Transform Wheel_BR;

    // Constants
    private double ConstWheelDist;
    private const double MaxSpeed = 5.0f; // m/s;
    private const double MaxTurnOmega = 0.1; // radians/s
    private const Int32 numObstacles = 3;
    private const Int32 inputSize = 6 + 3 * numObstacles;
    private const Int32 outputSize = 2;
    private const Int32 hiddenSize = 20; // random number. No idea.
    private const Int32 evaluationSamples = 1;
    private const double trainSimTime = 10.0f; // seconds
    private double elapsedTrainTime = 0.0;

    // Input
    private CarState _carState = new CarState();
    private List<ObstacleState> _obstacleStates = new List<ObstacleState>(numObstacles);

    // Output
    private CarInput _carInput = new CarInput();

    // Matrices
    // 3 Hidden layers for deeper learning. Don't have much reasoning.
    Layers resultLayers;
    bool trained = false;

    // Training parameters
    // How many total CEM iterations
    public static Int32 cemIterations = 100;
    // How many gaussian samples in each CEM iteration
    public static Int32 cemBatchSize = 100;
    // What percentage of cem samples are used to fit the gaussian for next iteration
    public static double cemEliteFrac = 0.15f;
    // Initial CEM gaussian uncertainty
    public static double cemInitStddev = 1.0f;
    // Scaling factor of how much extra noise to add each iteration (noise_factor/iteration_number noise is added to std.dev.)
    public static double cemNoiseFactor = 1.0f;

    // Start is called before the first frame update
    void Start()
    {
        ConstWheelDist = (Wheel_FL.position - Wheel_BL.position).magnitude;

        _carState.position = transform.position;
        _carState.turnTheta = 0;
        _carState.speed = 0;
        _carState.goalPosition = new Vector2(Goal.position.x, Goal.position.y);

        ObstacleState o1 = new ObstacleState();
        o1.position = new Vector2(0, 0);
        o1.radius = 5;
        _obstacleStates.Add(o1);

        ObstacleState o2 = new ObstacleState();
        o2.position = new Vector2(20, 0);
        o2.radius = 5;
        _obstacleStates.Add(o2);

        ObstacleState o3 = new ObstacleState();
        o3.position = new Vector2(-20, 0);
        o3.radius = 5;
        _obstacleStates.Add(o3);

        resultLayers = TrainRearAxelBicycleModel();
        trained = true;
        transform.position = _carState.position;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (trained)
        {
            if (elapsedTrainTime <= trainSimTime)
            {
                elapsedTrainTime += Time.deltaTime;
                RunModelSingleStep(in _carState, in _obstacleStates, ref _carInput, in resultLayers);
                UpdateStates(ref _carState, ref _obstacleStates, ref _carInput);
            }
        }
    }


    /***** * * * * * CEM Helpers * * * * * *****/
    private Layers TrainRearAxelBicycleModel()
    {
        Int32 numElite = (Int32)(cemBatchSize * cemEliteFrac);

        // Initialize our base set of layers.
        // This value will be iterated on each simulation iteration and
        // is the basis for the current iteration's batches
        Layers bestLayers = new Layers();
        bestLayers.Init(inputSize, hiddenSize, outputSize);
        bestLayers.SetStdDev(cemInitStddev);
        bestLayers.SetMean(0);

        // Run through our CEM
        for (Int32 i = 0; i < cemIterations; i++)
        {
            // Run batches based off of our current 'bestLayers' and store them
            // on order to find the elite results.
            List<Layers> batchLayers = new List<Layers>(cemBatchSize);
            List<double> batchRewards = new List<double>(cemBatchSize);
            for (Int32 j = 0; j < cemBatchSize; j++)
            {
                // Init a set of layers off of our current best set.
                Layers batchLayer = new Layers();
                batchLayer.Init(bestLayers, (float)cemInitStddev);

                // Calculate the reward for this batch and store it.
                batchRewards.Add(Reward(batchLayer));
                batchLayers.Add(batchLayer);
            }

            double meanReward = batchRewards.Average();

            // We now have our run-throughs, keep the best elite and set our layers back up.
            Int32[] indexArray = batchRewards
                .Select((value, index) => new { value, index })
                .OrderByDescending(item => item.value)
                .Take(numElite)
                .Select(item => item.index)
                .ToArray();

            // Store the mean and stdDev of our elite layers into our best layer.
            BatchMeanAndStdDev(in batchLayers, in indexArray, ref bestLayers);
            bestLayers.AddNoise(cemNoiseFactor / (i + 1));

            Debug.Log("Iteration: " + i + "\tmeanReward: " + meanReward + "\t Reward(CurrMatrix): " + Reward(bestLayers));
        }
        UnityEngine.Debug.Log("done");
        return bestLayers;
    }

    private void BatchMeanAndStdDev(in List<Layers> layers, in Int32[] indexArray, ref Layers outLayer)
    {
        for (Int32 i = 0; i < outLayer.Matrices.Count; i++)
        {
            for (Int32 j = 0; j < outLayer.Matrices[i].Length; j++)
            {
                for (Int32 k = 0; k < outLayer.Matrices[i][j].Length; k++)
                {
                    double sum = 0;
                    for (Int32 m = 0; m < indexArray.Length; m++)
                    {
                        sum += layers[indexArray[m]].Matrices[i][j][k];
                    }
                    sum /= (double)(indexArray.Length);
                    outLayer.Matrices[i][j][k] = sum;
                }
            }
        }

        for (Int32 i = 0; i < outLayer.Biases.Count; i++)
        {
            for (Int32 j = 0; j < outLayer.Biases[i].Length; j++)
            {
                double sum = 0;
                for (Int32 m = 0; m < indexArray.Length; m++)
                {
                    sum += layers[indexArray[m]].Biases[i][j];
                }
                sum /= (double)(indexArray.Length);
                outLayer.Biases[i][j] = sum;
            }
        }

        for (Int32 i = 0; i < outLayer.Matrices.Count; i++)
        {
            for (Int32 j = 0; j < outLayer.Matrices[i].Length; j++)
            {
                for (Int32 k = 0; k < outLayer.Matrices[i][j].Length; k++)
                {
                    double sum = 0;
                    for (Int32 m = 0; m < indexArray.Length; m++)
                    {
                        sum += (layers[indexArray[m]].Matrices[i][j][k] - outLayer.Matrices[i][j][k]) * (layers[indexArray[m]].Matrices[i][j][k] - outLayer.Matrices[i][j][k]);
                    }
                    sum /= (double)(indexArray.Length - 1);
                    outLayer.StdDevs[i][j][k] = Math.Sqrt(sum);
                }
            }
        }

        for (Int32 i = 0; i < outLayer.Biases.Count; i++)
        {
            for (Int32 j = 0; j < outLayer.Biases[i].Length; j++)
            {
                double sum = 0;
                for (Int32 m = 0; m < indexArray.Length; m++)
                {
                    sum += (layers[indexArray[m]].Biases[i][j] - outLayer.Biases[i][j]) * (layers[indexArray[m]].Biases[i][j] - outLayer.Biases[i][j]);
                }
                sum /= (double)(indexArray.Length - 1);
                outLayer.StdDevBiases[i][j] = Math.Sqrt(sum);
            }
        }

    }

    private void RunModelSingleStep(in CarState inCarState, in List<ObstacleState> inObstacleStates, ref CarInput outCarInput, in Layers layers)
    {
        double[][] inputArray = new double[inputSize][];
        for (Int32 i = 0; i < inputArray.Length; i++)
        {
            inputArray[i] = new double[1];
        }
        
        inputArray[0][0] = inCarState.position.x;
        inputArray[1][0] = inCarState.position.y;
        inputArray[2][0] = inCarState.turnTheta;
        inputArray[3][0] = inCarState.speed;
        inputArray[4][0] = inCarState.goalPosition.x;
        inputArray[5][0] = inCarState.goalPosition.y;

        Int32 obstacleIndex = 0;
        for (Int32 i = 6; i < 6 + numObstacles * 3; i += 3)
        {
            inputArray[i][0] = inObstacleStates[obstacleIndex].position.x;
            inputArray[i + 1][0] = inObstacleStates[obstacleIndex].position.y;
            inputArray[i + 2][0] = inObstacleStates[obstacleIndex].radius;

            obstacleIndex++;
        }


        // Hidden Layer 1
        double[][] result1 = new double[layers.Matrices[0].Length][];
        for (Int32 i = 0; i < result1.Length; i++)
        {
            result1[i] = new double[inputArray[0].Length];
        }
        for (Int32 i = 0; i < layers.Matrices[0].Length; i++)
        {
            for (Int32 j = 0; j < inputArray[0].Length; j++)
            {
                result1[i][j] = 0;
                for (Int32 k = 0; k < layers.Matrices[0][i].Length; k++)
                {
                    result1[i][j] += layers.Matrices[0][i][k] * inputArray[k][j];
                }
                result1[i][j] += layers.Biases[0][i];

                result1[i][j] = result1[i][j] * Convert.ToInt32(result1[i][j] > 0) + 0.1f * result1[i][j] * Convert.ToInt32(result1[i][j] < 0); // Relu
            }
        }
        // result1 is hiddenSize x 1

        // Hidden Layer 2
        double[][] result2 = new double[layers.Matrices[1].Length][];
        for (Int32 i = 0; i < result2.Length; i++)
        {
            result2[i] = new double[result1[i].Length];
        }
        for (Int32 i = 0; i < layers.Matrices[1].Length; i++)
        {
            for (Int32 j = 0; j < result1[i].Length; j++)
            {
                result2[i][j] = 0;
                for (Int32 k = 0; k < layers.Matrices[1][i].Length; k++)
                {
                    result2[i][j] += layers.Matrices[1][i][k] * result1[k][j];
                }
                result2[i][j] += layers.Biases[1][i];

                result2[i][j] = result2[i][j] * Convert.ToInt32(result2[i][j] > 0) + 0.1f * result2[i][j] * Convert.ToInt32(result2[i][j] < 0); // Relu
            }
        }
        // result2 is hiddenSize x 1


        // Hidden Layer 3
        double[][] result3 = new double[layers.Matrices[2].Length][];
        for (Int32 i = 0; i < result3.Length; i++)
        {
            result3[i] = new double[result2[i].Length];
        }
        for (Int32 i = 0; i < layers.Matrices[2].Length; i++)
        {
            for (Int32 j = 0; j < result2[i].Length; j++)
            {
                result3[i][j] = 0;
                for (Int32 k = 0; k < layers.Matrices[2][i].Length; k++)
                {
                    result3[i][j] += layers.Matrices[2][i][k] * result2[k][j];
                }
                result3[i][j] += layers.Biases[2][i];

                result3[i][j] = result3[i][j] * Convert.ToInt32(result3[i][j] > 0) + 0.1f * result3[i][j] * Convert.ToInt32(result3[i][j] < 0); // Relu
            }
        }
        // result3 is hiddenSize x 1


        // Output Layer
        double[][] result4 = new double[layers.Matrices[3].Length][];
        for (Int32 i = 0; i < result4.Length; i++)
        {
            result4[i] = new double[result3[i].Length];
        }
        for (Int32 i = 0; i < layers.Matrices[3].Length; i++)
        {
            for (Int32 j = 0; j < result3[0].Length; j++)
            {
                result4[i][j] = 0;
                for (Int32 k = 0; k < layers.Matrices[3][i].Length; k++)
                {
                    result4[i][j] += layers.Matrices[3][i][k] * result3[k][j];
                }
                result4[i][j] += layers.Biases[3][i];

                result4[i][j] = result4[i][j] * Convert.ToInt32(result4[i][j] > 0) + 0.1f * result4[i][j] * Convert.ToInt32(result4[i][j] < 0); // Relu
            }
        }
        // result4 is 2 x 1


        // Final results
        outCarInput.relativeTurnTheta = result4[0][0];
        outCarInput.relativeSpeed = result4[1][0];
    }

    List<ModelStepResult> RunModelFully(in Layers layers)
    {
        Int32 simIterations = (Int32)(trainSimTime / Time.deltaTime);
        List<ModelStepResult> modelStepResults = new List<ModelStepResult>(simIterations + 1);

        ModelStepResult initModelStep;
        initModelStep.carState = _carState;
        initModelStep.obstacleStates = _obstacleStates;
        initModelStep.carInput = _carInput;
        modelStepResults.Add(initModelStep);

        CarInput carInput = _carInput;
        CarState carState = _carState;
        List<ObstacleState> obstacleStates = _obstacleStates;
        for (Int32 i = 0; i < simIterations; i++)
        {
            RunModelSingleStep(in carState, in obstacleStates, ref carInput, in layers);

            UpdateStates(ref carState, ref obstacleStates, ref carInput);

            ModelStepResult modelStep;
            modelStep.carState = carState;
            modelStep.carInput = carInput;
            modelStep.obstacleStates = obstacleStates;
            modelStepResults.Add(modelStep);
        }

        return modelStepResults;
    }

    private double Reward(in Layers layers)
    {
        double totalReward = 0;
        transform.position = _carState.position;
        List<ModelStepResult> modelStepResults = RunModelFully(in layers);

        // Tabulate scoring for entire sequence of calculated actions

        double initDist = (_carState.position - _carState.goalPosition).magnitude;
        double dist = 0;
        // Car dist scoring
        for (Int32 j = 1; j < modelStepResults.Count; j++)
        {
            CarState state = modelStepResults[j - 1].carState;
            CarInput input = modelStepResults[j].carInput;
            dist = (state.position - state.goalPosition).magnitude;

            // Car scoring
            totalReward -= dist - initDist;
            totalReward -= 1*Math.Abs(input.relativeSpeed);
            totalReward -= 1*Math.Abs(input.relativeTurnTheta);

            // Obstacle scoring
            List<ObstacleState> obstacleStates = modelStepResults[j - 1].obstacleStates;
            for (Int32 k = 0; k < obstacleStates.Count; k++)
            {
                // Huge penalty for being within an obstacle
                totalReward -= (state.position - obstacleStates[k].position).magnitude < obstacleStates[k].radius ? 200000.0f : 0;
            }
        }

        totalReward += dist < initDist ? 10000 : 0;
        totalReward += dist < 30 ? 20000.0f : 0;
        totalReward += dist < 15 ? 30000.0f : 0;
        totalReward += dist < 10 ? 50000.0f : 0;
        totalReward += dist < 5f && modelStepResults[modelStepResults.Count - 1].carState.speed < 1.0f ? 200000.0f : 0;

        return totalReward / (double)evaluationSamples;
    }

    private void UpdateStates(ref CarState outCarState, ref List<ObstacleState> outObstacleStates, ref CarInput inCarInput)
    {
        inCarInput.relativeSpeed = Mathf.Clamp((float)inCarInput.relativeSpeed, -(float)MaxSpeed, (float)MaxSpeed);
        inCarInput.relativeTurnTheta = Mathf.Clamp((float)inCarInput.relativeTurnTheta, -(float)MaxTurnOmega, (float)MaxTurnOmega);

        // Update based off of neural network output
        outCarState.turnTheta += inCarInput.relativeTurnTheta * Time.deltaTime;
        outCarState.speed += inCarInput.relativeSpeed * Time.deltaTime;

        // Update position of model
        Vector2 currHeading;
        currHeading.x = (float)outCarState.speed * (float)Math.Sin(outCarState.turnTheta);
        currHeading.y = (float)outCarState.speed * (float)Math.Cos(outCarState.turnTheta);
        transform.position = transform.position + (currHeading.x * transform.right + currHeading.y * transform.up) * Time.deltaTime;
        outCarState.position = transform.position;

        // Update render of model
        transform.Rotate(0, 0, -(float)outCarState.turnTheta);
        Wheel_FL.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * -(float)outCarState.turnTheta);
        Wheel_FR.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * -(float)outCarState.turnTheta);
    }
}
