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

	//Matrix descriptors
	Double[][] inputData;
	Double[][] outputData;
	int[] networkDescription;
	double complexityCostLambda = 0.0001;
	ArrayList<Double[][]> weights = new ArrayList<Double[][]>();
	ArrayList<Double[][]> gradients = new ArrayList<Double[][]>();
	
	//Normalization Helper
	double[] normalizationFactors;
	
	//Results
	Double[][] yHat;
	double cost;
	
	public static void main(String[] args)
	{
		System.out.println("Begin matrix initialization...");
//		int[] networkDescription = {729,12,6,3,1};
		int[] networkDescription = {2,3,1};
		Double[][] inputData = {{3.,5.}, {5.,1.}, {10.,2.}};
		Double[][] outputData = {{.75}, {.82}, {.93}};
//		Double[][] inputData = {{0.1,0.1},{0.45,0.1},{0.1,0.2},{0.2,0.2},{0.35,0.2},{0.75,0.2},{0.3,0.3},{0.55,0.3},{0.7,0.3},{0.85,0.3},{0.05,0.4},{0.35,0.4},{0.45,0.4},{0.7,0.4},{0.9,0.4},{0.25,0.5},{0.6,0.5},{0.15,0.6},{0.35,0.6},{0.85,0.6},{0.05,0.7},{0.15,0.6},{0.2,0.8},{0.25,0.7},{0.45,0.8},{0.55,0.7},{0.55,0.9},{0.7,0.7},{0.7,0.9},{0.85,0.9},{0.9,0.7}};
//		Double[][] outputData = {{0.},{0.},{0.},{0.},{0.},{0.},{1.},{0.},{1.},{0.},{0.},{1.},{1.},{1.},{0.},{1.},{1.},{0.},{1.},{0.},{0.},{0.},{0.},{1.},{0.},{1.},{0.},{1.},{0.},{0.},{0.}};
		
		ImageHelper helper = new ImageHelper("data\\");
		
		
		
//		Double[][] inputData = {
//				helper.extractBytes("eight1.png"),
//				helper.extractBytes("eight2.png"),
//				helper.extractBytes("one1.png"),
//				helper.extractBytes("five1.png"),
//				helper.extractBytes("eight3.png"),
//				helper.extractBytes("eight5.png"),
//				helper.extractBytes("eight6.png"),
//				helper.extractBytes("nine1.png"),
//				helper.extractBytes("zero1.png"),
//				helper.extractBytes("eight7.png"),
//				helper.extractBytes("eight8.png"),
//				helper.extractBytes("eight10.png"),
//				helper.extractBytes("eight11.png"),
//				helper.extractBytes("eight12.png"),
//				helper.extractBytes("eight14.png"),
//				helper.extractBytes("four2.png"),
//				helper.extractBytes("four3.png"),
//				helper.extractBytes("eight4.png"),
//				helper.extractBytes("four1.png"),
//				helper.extractBytes("black.png"),
//				helper.extractBytes("six2.png"),
//				helper.extractBytes("six3.png"),
//				helper.extractBytes("six4.png"),
//				helper.extractBytes("six5.png")
//		};
//		Double[][] outputData = {
//				{1.},
//				{1.},
//				{0.},
//				{0.},
//				{1.},
//				{1.},
//				{1.},
//				{0.},
//				{0.},
//				{1.},
//				{1.},
//				{1.},
//				{1.},
//				{1.},
//				{1.},
//				{0.},
//				{0.},
//				{1.},
//				{0.},
//				{0.},
//				{0.},
//				{0.},
//				{0.},
//				{0.}
//		};
		
		Double[][] testData = {
//				helper.extractBytes("eight16.png"),
//				helper.extractBytes("six1.png")
				{8.,3.}
		};
		
		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "Dave_Test_2_3_1.txt");
//		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "2_729.11.7.5.1.txt", "2_729.11.7.5.1.txt");

//		NN.saveWeights();
		testData = NN.normalizeMatrix(testData);
		
		NN.calculateForwardProp();
		double cost = NN.calculateCostFunction(NN.outputData, NN.yHat);
		
		NN.calculateCostFunctionPrimes();
		printMatrix(NN.yHat);
		
		double cost2 = NN.cost;

		double[] unraveledWeights = NN.unravel(NN.weights);
		double[] unraveledGradient = NN.unravel(NN.gradients);
		try {

			do
			{
				LBFGS.lbfgs(NN.numberOfVariables, 300, unraveledWeights, cost2, unraveledGradient, false, NN.diag, NN.iprint, 1.0e-4, 1.0e-17, NN.iflag);
				NN.reravel(unraveledWeights, NN.weights);
				NN.calculateForwardProp();
				NN.calculateCostFunctionPrimes();
				NN.calculateCostFunction(NN.outputData, NN.yHat);
				cost2 = NN.cost;
				unraveledGradient = NN.unravel(NN.gradients);
				NN.printMatrix(NN.yHat);
		
			} while(NN.iflag[0] != 0);

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("error");
		}
		
		NN.saveWeights();
		NN.inputData = testData;
		NN.calculateForwardProp();
		System.out.println("******");
		printMatrix(NN.yHat);
	}
	
	public NeuralNet()
	{
		iprint [ 1 -1] = 1;
		iprint [ 2 -1] = 0;
		iprint [ 0] = 1;

		iflag[0] = 0;
		icall = 0;
	}

	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData)
	{
		//Setup for LBFGS algorithm
		this();
		
		//Begin initialization of network
		this.inputData = inputData;
		this.outputData = outputData;
		this.networkDescription = networkDescription;
		

		//Normalize Weights
		normalizeMatrix(this.inputData);
		
		//Initialize Weights
		initializeValuesInMatrix(true, weights);
		
		//Initialize Gradients
		initializeValuesInMatrix(false, gradients);
		
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			numberOfVariables += (networkDescription[i] * networkDescription[i+1]);
		}
		diag = new double [ numberOfVariables ];
	}
	
	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData, String saveFile)
	{
		this(networkDescription, inputData, outputData);
		//Begin initialization of network
		this.saveFile = saveFile;
		
	}
	
	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData, String loadFile, String saveFile)
	{
		
		//Setup for LBFGS algorithm
		this();
		
		//Begin initialization of network
		this.inputData = inputData;
		this.outputData = outputData;
		this.networkDescription = networkDescription;
		this.loadFile = loadFile;
		this.saveFile = saveFile;
		File weightInitiatorPath = new File(filePath + loadFile);
		
		//Normalize Weights
		normalizeMatrix(this.inputData);
		
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
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		//Initialize Gradients
		initializeValuesInMatrix(false, gradients);
		
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			numberOfVariables += (networkDescription[i] * networkDescription[i+1]);
		}
		diag = new double [ numberOfVariables ];
	}
	
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
			printMatrix(tempLayer);
		}
	}
	
	public void calculateForwardProp()
	{
		Double[][] a = inputData;
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] z = new Double[a.length][weights.get(i)[0].length];
			z = NeuralMatrix.multiply(a, weights.get(i));
			a = NeuralMatrix.applySigmoid(z);
		}
		yHat = a;
	}

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
	
	public void calculateCostFunctionPrimes()
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
		
	public static void printMatrix(Double[][] a)
	{
        for (int i = 0; i < a.length; i++)
        {
            for (int j = 0; j < a[0].length; j++)
            {
                System.out.print(a[i][j] + ", ");
            }
            System.out.println();
        }
	}
	
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
		    System.out.println("******************File save completed");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	

}
