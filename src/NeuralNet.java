import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class NeuralNet {
	//LBFGS variables
	int iflag[] = {0};
	int iprint[] = new int[2];
	int icall;
	double diag[];
	int numberOfVariables=0;
	String filePath = "C:\\NeuralNet\\";
	String loadFile = "test.txt";
	String saveFile = "test.txt";
	double thresholdError = 1.0e-4;

	//Matrix descriptors
	Double[][] inputData;
	Double[][] outputData;
	int[] networkDescription;
	double complexityCostLambda = 0.0001;
	ArrayList<Double[][]> weights = new ArrayList<Double[][]>();
	ArrayList<Double[][]> gradients = new ArrayList<Double[][]>();
	ArrayList<Double[][]> activationFactors = new ArrayList<Double[][]>();
	ArrayList<Double[][]> errorFactors = new ArrayList<Double[][]>();
	
	//Normalization Helper
	double[] normalizationFactors;
	
	//Results
	Double[][] yHat;
	double cost;
	
	public static void main(String[] args)
	{
		//We need to define the hyper parameters of our neural network
		System.out.println("Begin matrix initialization...");
		int[] networkDescription = {729,90,1};
		
		//Now we want to define our input data.  We use a helper class in order to help extract the information
		ImageHelper helper = new ImageHelper("data\\");
		
		Double[][] inputData = {
				helper.extractBytes("eight1.png"),
				helper.extractBytes("eight2.png"),
				helper.extractBytes("eight3.png"),
				helper.extractBytes("eight4.png"),
				helper.extractBytes("eight5.png"),
				helper.extractBytes("eight6.png"),
				helper.extractBytes("eight7.png"),
				helper.extractBytes("eight8.png"),
				helper.extractBytes("eight10.png"),
				helper.extractBytes("eight11.png"),
				helper.extractBytes("eight12.png"),
				helper.extractBytes("eight14.png"),
				helper.extractBytes("nine1.png"),
				helper.extractBytes("zero1.png"),
				helper.extractBytes("one1.png"),
				helper.extractBytes("four1.png"),
				helper.extractBytes("five1.png"),
				helper.extractBytes("four2.png"),
				helper.extractBytes("four3.png"),
				helper.extractBytes("black.png"),
				helper.extractBytes("six2.png"),
				helper.extractBytes("six3.png"),
				helper.extractBytes("six4.png"),
				helper.extractBytes("six5.png"),
				helper.extractBytes("seven5.png")
		};
		Double[][] outputData = {
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.},
				{0.}
		};
		
		//This is the test data we want to measure our results against
		Double[][] testData = {
				helper.extractBytes("eight16.png"),
				helper.extractBytes("eight17.png"),
				helper.extractBytes("eight18.png"),
				helper.extractBytes("eight19.png"),
				helper.extractBytes("eight20.png"),
				helper.extractBytes("eight21.png"),
				helper.extractBytes("eight22.png"),
				helper.extractBytes("eight26.png"),
				helper.extractBytes("nine3.png"),
				helper.extractBytes("nine4.png"),
				helper.extractBytes("nine7.png"),
				helper.extractBytes("seven7.png"),
				helper.extractBytes("seven6.png")
		};
		
		//Initialize Network and normalize Testing data (training data is normalized in the initialization right now)
		//Our neural net is going to be normalized during creation
		//The idea of our normalization factors created during normalization is so we can multiple 
		//our normalization factors agsinst our results to receive our final answer.
		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "fullImageTest_20170529.txt", "fullImageTest_20170529.txt");
		
		//Since we've already trained some data, we want to load the pre-computed weights. 
		//This can only be done if you have a properly constructed file at the given location: C:\\NeuralNet\\<yourFileName>
//		NN.loadFile();
		
		//Finally we can train our dataset
		NN.trainDataset();
		
		//Training now should be complete.  At this point we can save our weights, which is our actual computed answer.
		System.out.println("Saving Weights...");
		NN.saveWeights();
		
		//Lets test our answer against our test data and see how we did.
		NN.inputData = testData;
		System.out.println("Calculating our guess...");
		NN.calculateForwardProp();
		System.out.println("******");
		NeuralMatrix.printMatrix(NN.yHat);
	}
	
	
	public void trainDataset()
	{
		//Begin first pass of the neural network
		calculateForwardProp();
		calculateCostFunction(yHat, outputData);
		
		//If we already have an acceptable answer, there is no need to run our training algorithm. 
		if(thresholdError > this.cost)
		{
			System.out.println("Error is within acceptable threshold - no training needed");
			System.out.println("Output of Neural Net");
			NeuralMatrix.printMatrix(this.yHat);
			return;
		}

		//Lets calculate our gradients or first derivative of our neural net function
		calculateCostFunctionGradient();
		NeuralMatrix.printMatrix(yHat);
		
		//Save some temp variables and begin unraveling the weights for back propagation
		double cost2 = this.cost;
		double[] unraveledWeights = unravel(weights);
		double[] unraveledGradient = unravel(gradients);
		System.out.println("Beginning Back propogation...");
		try {

			do
			{
				//This is the heart of our back propogation.  The LBFGS algorithm takes in a set of gradients, current cost, and other items
				//in order to properly calculate the next step we should take.
				LBFGS.lbfgs(numberOfVariables, 300, unraveledWeights, cost2, unraveledGradient, false, diag, iprint, thresholdError, 1.0e-17, iflag);
				
				//Lets place our new weights back into the proper place and recalculate our gradients and cost
				reravel(unraveledWeights, weights);
				calculateForwardProp();
				calculateCostFunctionGradient();
				calculateCostFunction(outputData, yHat);
				cost2 = this.cost;
				
				//Prepare gradients for next pass
				unraveledGradient = unravel(gradients);
				NeuralMatrix.printMatrix(yHat);
		
			} while(iflag[0] != 0);

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("An error has occurred during back propogation and training");
		}
		System.out.println("Output of Neural Net");
	}
	
	public NeuralNet()
	{
		iprint [ 1 -1] = 1;
		iprint [ 2 -1] = 0;
		iprint [ 0] = 1;

		iflag[0] = 0;
		icall = 0;
	}
	
	public NeuralNet(int[] networkDescription)
	{
		//Setup for LBFGS algorithm
		this();
		
		//Begin Initialization of network
		this.networkDescription = networkDescription;
		
		//Initialize Weights
		initializeValuesInMatrix(true, weights);
		
		//Initialize Gradients
		initializeValuesInMatrix(false, gradients);
		
		//Initialize Error Factors
		initializeValuesInMatrix(false, errorFactors);
		
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			numberOfVariables += (networkDescription[i] * networkDescription[i+1]);
		}
		diag = new double [ numberOfVariables ];
		
	}

	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData)
	{
		//Call lower constructor
		this(networkDescription);
		
		//Begin initialization of network
		this.inputData = inputData;
		this.outputData = outputData;
		
		//Normalize Weights
		normalizeMatrix(this.inputData);
		
	}
	
	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData, String saveFile)
	{
		//Setup for LBFGS algorithm and initialization of network
		this(networkDescription, inputData, outputData);
		
		//Begin initialization of network
		this.saveFile = saveFile;
		
	}
	
	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData, String saveFile, String loadFile)
	{
		//Setup for LBFGS algorithm and initialization of network
		this(networkDescription, inputData, outputData);
		
		//Initialize save and load file locations
		this.loadFile = loadFile;
		this.saveFile = saveFile;
	}
	
	//When we normalize our input data, we want to normalize it per column rather than overall since each
	//column has the possibility of representing its own piece of distinct information and columns do not
	//necessarily relate to each other in any way
	public Double[][] normalizeMatrix(Double[][] normalizingMatrix)
	{
		//Find Max Value. If already found, then skip
		double[] maxOfColumn;
		if(normalizationFactors == null)
		{
			maxOfColumn = new double[normalizingMatrix[0].length];
			for(int i = 0; i < normalizingMatrix.length; i++)
			{
				for(int j = 0; j < normalizingMatrix[i].length; j++)
				{
					if(maxOfColumn[j] < normalizingMatrix[i][j])
						maxOfColumn[j] = normalizingMatrix[i][j];
				}
			}
			normalizationFactors = maxOfColumn;
		}     
		
		for(int i = 0; i < normalizingMatrix.length; i++)
		{
			for(int j = 0; j < normalizingMatrix[i].length; j++)
			{
				if(normalizationFactors[j] == 0)
					normalizingMatrix[i][j] = 0.0;
				else
					normalizingMatrix[i][j] = normalizingMatrix[i][j]/normalizationFactors[j];
			}
		}
		
		return normalizingMatrix;
	}
	
	
	private void initializeValuesInMatrix(boolean isRandom, ArrayList<Double[][]> initializeItems)
	{
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] tempLayer = new Double[networkDescription[i]][networkDescription[i+1]];
			for(int x = 0; x < networkDescription[i]; x++)
			{
				for(int y = 0; y < networkDescription[i+1]; y++)
				{
					if(isRandom)
					{
						double randomNum = Math.random()*2-1;
						int curWeightInt = ((int)(randomNum * 1000));
						double curWeight = curWeightInt/1000.0;
						tempLayer[x][y] = curWeight;
					}
					else
					{
						tempLayer[x][y] = 0.0;
					}
				}
			}
			initializeItems.add(tempLayer);
		}
	}
	
	//This is how we get the answer of our neural net.  It is simply a set of matrix manipulation and storage
	public void calculateForwardProp()
	{
		Double[][] a = inputData;
		activationFactors.add(0, a);
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] z = new Double[a.length][weights.get(i)[0].length];
			z = NeuralMatrix.multiply(a, weights.get(i));
			a = NeuralMatrix.applySigmoid(z);
			activationFactors.add(i+1, a);
		}
		yHat = a;
	}

	//Defines the cost function of our neural net and outputs the cost of our neural net
	public double calculateCostFunction(Double[][] Y, Double[][] yHat)
	{
		//Calculate weight complexity
		double complexityCost = this.calculateWeightComplexity();
		
		//Calculate the yHat - Y, square it, then half it (1/2) (y-yHat)^2
		Double[][] costResultMatrix = NeuralMatrix.subtract(Y, yHat);
		costResultMatrix = NeuralMatrix.multiplyVector(costResultMatrix, costResultMatrix);
		double answerCost = NeuralMatrix.sumVector(costResultMatrix)/2;
		
		//Sum our MatrixCost and our weightCost
		this.cost = answerCost + complexityCost;
		
		return this.cost;
	}
	
	//Calculate the exact gradient for our NeuralNetwork
	//IT WORKS!!
	public void calculateCostFunctionGradient()
	{
		for(int i = networkDescription.length - 1; i >= 0; i--)
		{
			Double[][] errorFactor;
			if(i == networkDescription.length - 1)
			{
				Double[][] diff = NeuralMatrix.subtract(yHat, outputData);
				Double[][] sigmoidGradient = NeuralMatrix.multiplyScalar(NeuralMatrix.subtractValue(1, activationFactors.get(i)), activationFactors.get(i));
				errorFactor = NeuralMatrix.multiplyVector(sigmoidGradient, diff);
				errorFactors.add(i, errorFactor);
			}
			else
			{
				Double[][] p1 = NeuralMatrix.subtractValue(1, activationFactors.get(i));
				Double[][] p2 = activationFactors.get(i);
				Double[][] sigmoidGradient = NeuralMatrix.multiplyScalar(p1, p2);
				
				Double[][] weightTranspose = NeuralMatrix.transpose(weights.get(i));
				Double[][] temp = NeuralMatrix.multiply(errorFactors.get(i+1), weightTranspose);
				System.out.println(temp.length + " x " + temp[0].length);
				System.out.println(sigmoidGradient.length + " x " + sigmoidGradient[0].length);
				errorFactor = NeuralMatrix.multiplyScalar(temp, sigmoidGradient);
				errorFactors.set(i, errorFactor);
				
				Double[][] activationFactorTranspose = NeuralMatrix.transpose(activationFactors.get(i));
				Double[][] tempErrorFactors = errorFactors.get(i+1);
				Double[][] gradientTempResult = NeuralMatrix.multiply(activationFactorTranspose, tempErrorFactors); 
				
				//Add regularization to our algorithm
				Double[][] regularization = NeuralMatrix.multiplyScalar(weights.get(i), (complexityCostLambda/2.0));
				
				Double[][] gradientResult = NeuralMatrix.add(gradientTempResult, regularization);
				gradients.set(i, gradientResult);
			}
		}
	}
	
	//Apply numerical gradient for now.  This can be used if the cost function, activation function, or neuron type changes
	public void calculateCostFunctionPrimesNumericalGradient()
	{
		double epsilon = .00000001;

		double[] unraveled = this.unravel(weights);
		double[] gradient = new double[unraveled.length];
		
		//We come into this method with varying weights from the original set sometimes.  Therefore, calculate the cost and store it.
		calculateForwardProp();
		this.calculateCostFunction(outputData, yHat);
		double origCost = this.cost;

		System.out.println("unraveled length: " + unraveled.length);
		for(int i = 0; i < unraveled.length; i++)
		{
			unraveled[i] = unraveled[i] + epsilon;
			this.reravel(unraveled, this.weights);
			calculateForwardProp();
			this.calculateCostFunction(outputData, yHat);
			double pCost = this.cost;

			gradient[i] = (pCost - origCost) / (epsilon);

			unraveled[i] = unraveled[i] - epsilon;
			this.reravel(unraveled, this.weights);
		}
		this.reravel(gradient, this.gradients);
	}
	
	
	//This method helps to apply regularization to the neural network.  
	//We say the network is complex if the squared sum of the weights is large.
	public double calculateWeightComplexity()
	{
		double weightComplexity = 0.0;
		for(int i = 0; i < this.weights.size(); i++)
		{
			Double[][] curWeightLayer = this.weights.get(i);
			for(int x = 0; x < curWeightLayer.length; x++)
			{
				for(int y = 0; y < curWeightLayer[0].length; y++)
				{
					weightComplexity += (curWeightLayer[x][y]) * (curWeightLayer[x][y]);
				}
			}
		}
		weightComplexity = weightComplexity * (complexityCostLambda/2.0);
		return weightComplexity;
	}

	//Helper method.  Takes in some 3D matrix and converts into a 1D matrix
	public double[] unravel(ArrayList<Double[][]> unravelItems)
	{
		ArrayList<Double> myList = new ArrayList<Double>();
		for(int i = 0; i < unravelItems.size(); i++)
		{
			Double[][] curWeightLayer = unravelItems.get(i);
			for(int x = 0; x < curWeightLayer.length; x++)
			{
				for(int y = 0; y < curWeightLayer[0].length; y++)
				{
					myList.add(curWeightLayer[x][y]);
				}
			}
		}
		
		Double[] unraveledList = new Double[myList.size()];
		unraveledList = myList.toArray(unraveledList);
		double[] d = new double[myList.size()];
		for(int i = 0; i < myList.size(); i++)
		{
			d[i] = unraveledList[i];
		}
		return d;
	}
	
	//Helper method.  Takes in a 1D matrix and converts it into the proper 3D matrix
	public void reravel(double[] d, ArrayList<Double[][]> reravelItem)
	{
		int dPosition = 0;
		for(int i = 0; i < reravelItem.size(); i++)
		{
			Double[][] curWeightLayer = reravelItem.get(i);
			for(int x = 0; x < curWeightLayer.length; x++)
			{
				for(int y = 0; y < curWeightLayer[0].length; y++)
				{
					curWeightLayer[x][y] = d[dPosition];
					dPosition++;
				}
			}
			reravelItem.set(i, curWeightLayer);
		}
		
	}
	
	//Save the weights to the proper save file
	public void saveWeights()
	{
		try(  PrintWriter out = new PrintWriter( this.filePath + this.saveFile )  ){
			String networkDescriptionString = "";
			for(int i = 0; i < networkDescription.length; i++)
			{
				networkDescriptionString += networkDescription[i] + " ";
			}
			
			ArrayList<String> weightsString = new ArrayList<String>();
			for(int i = 0; i < weights.size(); i++)
			{
				Double[][] d = weights.get(i);
				for(int x = 0; x < d.length; x++)
				{
					Double[] dx = d[x];
					String curWeightLine = "";
					for(int y = 0; y < dx.length; y++)
					{
						curWeightLine += dx[y] + " ";
					}
					weightsString.add(curWeightLine);
				}
				weightsString.add("----");
			}
			
			out.println(networkDescriptionString);
			out.println("----");
		    for(String s : weightsString)
		    {
		    	out.println(s);
		    }
		    out.println("EOF");
		    System.out.println("File save completed");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	//Save the weights to the proper save file
	public void saveGradients()
	{
		try(  PrintWriter out = new PrintWriter( this.filePath + "gradients" + this.saveFile )  ){
			String networkDescriptionString = "";
			for(int i = 0; i < networkDescription.length; i++)
			{
				networkDescriptionString += networkDescription[i] + " ";
			}
			
			ArrayList<String> weightsString = new ArrayList<String>();
			for(int i = 0; i < gradients.size(); i++)
			{
				Double[][] d = gradients.get(i);
				for(int x = 0; x < d.length; x++)
				{
					Double[] dx = d[x];
					String curWeightLine = "";
					for(int y = 0; y < dx.length; y++)
					{
						curWeightLine += dx[y] + " ";
					}
					weightsString.add(curWeightLine);
				}
				weightsString.add("----");
			}
			
			out.println(networkDescriptionString);
			out.println("----");
		    for(String s : weightsString)
		    {
		    	out.println(s);
		    }
		    out.println("EOF");
		    System.out.println("File save completed");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	
	//Load the weights from the specified file
	public void loadFile()
	{
		weights = new ArrayList<Double[][]>();
		File weightInitiatorPath = new File(filePath + loadFile);
		
		//Read from file and initialize weights
		try(BufferedReader br = new BufferedReader(new FileReader(weightInitiatorPath))) {
			StringBuilder sb = new StringBuilder();
		    String line = br.readLine();
		    String[] splitter = line.split(" ");
		    networkDescription = new int[splitter.length];
		    for(int i = 0; i < splitter.length; i++)
		    {
		    	networkDescription[i] = Integer.parseInt(splitter[i]);
		    }
		    line = br.readLine();
		    line = br.readLine();
		    int counter = 0;
		    int seperatorsFound = 0;
		    while (!line.equals("EOF") && line != null) {
		    	
		    	Double[][] tempWeights = new Double[networkDescription[seperatorsFound]][networkDescription[seperatorsFound+1]];
		    	System.out.println("Matrix Initialized: " + tempWeights.length + " x " + tempWeights[0].length);
		    	
		    	while (!line.equals("EOF") && !line.equals("----") && line != null) {
				    String[] matrixSplitter = line.split(" ");
				    
				    for(int i = 0; i < matrixSplitter.length; i++)
				    {
				    	tempWeights[counter][i] = Double.parseDouble(matrixSplitter[i]);
				    }
				    counter++;
					

			        line = br.readLine();
			        if(line.equals("----"))
			        {
			        	seperatorsFound++;
			        	weights.add(tempWeights);
			        }

		    	}
		    	counter=0;
		        line = br.readLine();
		    }
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	

}
