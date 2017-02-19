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
	ArrayList<Double[][]> weights = new ArrayList<Double[][]>();
	ArrayList<Double[][]> gradients = new ArrayList<Double[][]>();
	
	//Results
	Double[][] yHat;
	double cost;
	
	public static void main(String[] args)
	{
		System.out.println("Begin matrix initialization...");
		int[] networkDescription = {729,11,7,5,1};
//		int[] networkDescription = {2,3,1};
//		Double[][] inputData = {{.3,.5,}, {.5,.1,}, {1.,.2,}};
//		Double[][] outputData = {{.75}, {.82}, {.93}};
//		Double[][] inputData = {{0.1,0.1},{0.45,0.1},{0.1,0.2},{0.2,0.2},{0.35,0.2},{0.75,0.2},{0.3,0.3},{0.55,0.3},{0.7,0.3},{0.85,0.3},{0.05,0.4},{0.35,0.4},{0.45,0.4},{0.7,0.4},{0.9,0.4},{0.25,0.5},{0.6,0.5},{0.15,0.6},{0.35,0.6},{0.85,0.6},{0.05,0.7},{0.15,0.6},{0.2,0.8},{0.25,0.7},{0.45,0.8},{0.55,0.7},{0.55,0.9},{0.7,0.7},{0.7,0.9},{0.85,0.9},{0.9,0.7}};
//		Double[][] outputData = {{0.},{0.},{0.},{0.},{0.},{0.},{1.},{0.},{1.},{0.},{0.},{1.},{1.},{1.},{0.},{1.},{1.},{0.},{1.},{0.},{0.},{0.},{0.},{1.},{0.},{1.},{0.},{1.},{0.},{0.},{0.}};
		
		ImageHelper helper = new ImageHelper("data\\");
		
		Double[][] inputData = {
				helper.extractBytes("eight1.png"),
				helper.extractBytes("eight2.png"),
				helper.extractBytes("one1.png"),
				helper.extractBytes("five1.png"),
				helper.extractBytes("eight3.png"),
				helper.extractBytes("eight5.png"),
				helper.extractBytes("eight6.png"),
				helper.extractBytes("nine1.png"),
				helper.extractBytes("zero1.png"),
				helper.extractBytes("eight7.png"),
				helper.extractBytes("eight8.png"),
				helper.extractBytes("eight10.png"),
				helper.extractBytes("eight11.png"),
				helper.extractBytes("eight12.png"),
				helper.extractBytes("eight14.png"),
				helper.extractBytes("four2.png"),
				helper.extractBytes("four3.png"),
				helper.extractBytes("eight4.png"),
				helper.extractBytes("four1.png")
		};
		Double[][] outputData = {
				{1.},
				{1.},
				{0.},
				{0.},
				{1.},
				{1.},
				{1.},
				{0.},
				{0.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{1.},
				{0.},
				{0.},
				{1.},
				{0.}
		};
		
		Double[][] testData = {
				helper.extractBytes("eight4.png"),
				helper.extractBytes("four1.png")
		};
		
//		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "2_729.11.7.5.1.txt", "2_729.11.7.5.1.txt");
		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "1_729.11.7.5.1.txt");

//		NN.saveWeights();
		
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
				LBFGS.lbfgs(NN.numberOfVariables, 300, unraveledWeights, cost2, unraveledGradient, false, NN.diag, NN.iprint, 1.0e-5, 1.0e-17, NN.iflag);
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

		iflag[0]=0;
		icall=0;
	}

	public NeuralNet(int[] networkDescription, Double[][] inputData, Double[][] outputData)
	{
		//Begin initialization of network
		this.inputData = inputData;
		this.outputData = outputData;
		this.networkDescription = networkDescription;
		
		//Initialize Weights
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] tempWeights = new Double[networkDescription[i]][networkDescription[i+1]];
			for(int x = 0; x < networkDescription[i]; x++)
			{
				for(int y = 0; y < networkDescription[i+1]; y++)
				{
					double randomNum = Math.random()*2-1;
					int curWeightInt = ((int)(randomNum * 1000));
					double curWeight = curWeightInt/1000.0;
					tempWeights[x][y] = curWeight;
				}
			}
			weights.add(tempWeights);
			printMatrix(tempWeights);
		}
		
		
		
		//Initialize Gradients
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] tempGradient = new Double[networkDescription[i]][networkDescription[i+1]];
			for(int x = 0; x < networkDescription[i]; x++)
			{
				for(int y = 0; y < networkDescription[i+1]; y++)
				{
					tempGradient[x][y] = 0.0;
				}
			}
			gradients.add(tempGradient);
//			printMatrix(tempGradient);
		}
		
		
		//Setup for LBFGS algorithm
		iprint [ 1 -1] = 1;
		iprint [ 2 -1] = 0;
		iprint [ 0] = 1;
		iflag[0]=0;
		icall=0;
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
		//Begin initialization of network
		this.inputData = inputData;
		this.outputData = outputData;
		this.networkDescription = networkDescription;
		
		File weightInitiatorPath = new File(filePath + loadFile);
		
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
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] tempGradient = new Double[networkDescription[i]][networkDescription[i+1]];
			for(int x = 0; x < networkDescription[i]; x++)
			{
				for(int y = 0; y < networkDescription[i+1]; y++)
				{
					tempGradient[x][y] = 0.0;
				}
			}
			gradients.add(tempGradient);
//			printMatrix(tempGradient);
		}
		
		
		//Setup for LBFGS algorithm
		iprint [ 1 -1] = 1;
		iprint [ 2 -1] = 0;
		iprint [ 0] = 1;
		iflag[0]=0;
		icall=0;
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			numberOfVariables += (networkDescription[i] * networkDescription[i+1]);
		}
		diag = new double [ numberOfVariables ];
	}
	
	public void calculateForwardProp()
	{
		Double[][] a = inputData;
		for(int i = 0; i < networkDescription.length - 1; i++)
		{
			Double[][] z = new Double[a.length][weights.get(i)[0].length];
			z = NeuralMatrix.multiply(a, weights.get(i));
			a = NeuralMatrix.applySigmoid(z);
//			printMatrix(z);
		}
//		printMatrix(a);
		yHat = a;
	}

	public double calculateCostFunction(Double[][] Y, Double[][] yHat)
	{
		
		Double[][] costResultMatrix = NeuralMatrix.subtract(Y, yHat);
		costResultMatrix = NeuralMatrix.multiplyVector(costResultMatrix, costResultMatrix);
		double costResult = NeuralMatrix.sumVector(costResultMatrix)/2;
		this.cost = costResult;
		
		return costResult;
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
			
/*			This was an old way of calculating the partial derivative.  Now we just choose a smaller 
 * 			delta, calculate the original cost at the top, and only move a weight by one delta.
 * 			This effectively halves the number of evaluations for calculating the cost function primes.
*/			
//			unraveled[i] = unraveled[i] - epsilon - epsilon;
//			this.reravel(unraveled, this.weights);
//			calculateForwardProp();
//			this.calculateCostFunction(outputData, yHat);
//			double nCost = this.cost;
//			gradient[i] = (pCost - nCost) / (2*epsilon);
			
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
		//Something doesn't work here if you have the file already existing.  Think we need
		//to first delete the old file then write the new file.
		try(  PrintWriter out = new PrintWriter( this.filePath + this.saveFile )  ){
			String networkDescriptionString = "";
			for(int i = 0; i < networkDescription.length; i++)
			{
				networkDescriptionString += networkDescription[i] + " ";
			}
			
//			ArrayList<Double[][]> weights = new ArrayList<Double[][]>();
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
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	

}
