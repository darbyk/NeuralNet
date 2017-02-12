import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork {
	
	int inputLayerSize;
	int outputLayerSize;
	int hiddenLayerSize;
	
	double[][] W1 = { { .427, 0.141, .486}, { .942, .583, .983} };
	double[][] W2 = { {.209}, { .666}, { .393} };
	
	//Starting Values
//	double[][] W1 = { { 0.47294868, 0.20886289, -0.38692276 }, { -1.62662175, -0.19100219, 0.58192548 } };
//	double[][] W2 = { { -1.67831991 }, { -0.81186678 }, { 2.3513771  } };

	
	//Answer
//	double[][] W1 = { { -0.35193307, 0.09626478, 1.90199972 }, { -1.7292198, -0.2605566, -0.86066199 } };
//	double[][] W2 = { { -3.33952974 }, { -1.26187598 }, { 4.9452339 } };
	
	
	double[][] X = {{.3,.5}, {.5,.1}, {.10,.2}};
	double[][] Y = {{.75}, {.82}, {.93}};
//	double[][] X = {{.2,1}, {.4,1}, {.6,1}, {.7,1}, {.8,1}, {.5,1}, {.25,1}, {.35,1}, {1,1}, {.88,1}};
//	double[][] Y = {{.04}, {.16}, {.36}, {.49}, {.64}, {.25}, {.0625}, {.1225}, {1},{.7744}};
	
	double[][] Z2;
	double[][] A2;
	double[][] Z3;
	double[][] yHat;
	double[][] D3;
	double[][] D2;
	double[][] djdw1 = new double[W1.length][W1[0].length];
	double[][] djdw2 = new double[W2.length][W2[0].length];;
	
	double cost;
	
	public static void main(String[] args)
	{
		System.out.println("Begin matrix initialization...");
		NeuralNetwork NN = new NeuralNetwork();
		
		NN.calculateForwardProp();
		NN.calculateCostFunctionPrimes();
		NN.calculateCostFunction(NN.Y, NN.yHat);
		
		System.out.println("**");
		printMatrix(NN.yHat);
		
		int iflag[] = {0};
		int iprint[] = new int[2];
		iprint [ 1 -1] = 1;
		iprint [ 2 -1] = 0;
		
		iprint [ 0] = 1;
		

		int icall=0;
		iflag[0]=0;

//		double[] yHatT = Matrix.transpose(NN.Y)[0];
//		double[] dsdw2t = Matrix.transpose(NN.djdw2)[0];
//		double[] nnw2 = Matrix.transpose(NN.W2)[0];
		double cost2 = NN.cost;
		
//		System.out.print("Cost: " + cost2 + "\t");
//		System.out.println();
//		System.out.println("z3");
//		NN.printMatrix(NN.Z3);

		System.out.println("djdw1");
		NN.printMatrix(NN.djdw1);
		System.out.println("djdw2");
		NN.printMatrix(NN.djdw2);
		
//		System.out.println("yhat");
//		NN.printMatrix(NN.yHat);
		double x [ ] , g [ ] , diag [ ] , w [ ];
		diag = new double [ 15 ];
		double[] unraveledWeights = NN.unravel(NN.W1, NN.W2);
		System.out.println();
		for(double i : unraveledWeights)
			System.out.print(i + ", ");
		
		double[] unraveledGradient = NN.unravel(NN.djdw1, NN.djdw2);
		
		try {
			
			do
			{
				
				LBFGS.lbfgs(9, 200, unraveledWeights, cost2, unraveledGradient, false, diag, iprint, 1.0e-7, 1.0e-15, iflag);
				NN.reravel(unraveledWeights);
				NN.calculateForwardProp();
				NN.calculateCostFunctionPrimes();
				NN.calculateCostFunction(NN.Y, NN.yHat);
				cost2 = NN.cost;
				unraveledGradient = NN.unravel(NN.djdw1, NN.djdw2);
				NN.printMatrix(NN.yHat);
			} while(iflag[0] != 0);
			System.out.println(NN.cost);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("error");
		}
		
		System.out.println();
		System.out.println();
//		double[][] tempX = {{.3, 1}, {.5, 1}, {.9, 1}};
//		NN.X = tempX;
//		NN.calculateForwardProp();
//		NN.printMatrix(NN.yHat);
		

	}
	
	public double[] unravel(double[][] a, double[][] b)
	{
		ArrayList<Double> myList = new ArrayList<Double>();
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < a[0].length; j++)
				myList.add(a[i][j]);
		}
		for(int i = 0; i < b.length; i++)
		{
			for(int j = 0; j < b[0].length; j++)
				myList.add(b[i][j]);
		}
		Double[] test = new Double[myList.size()];
		test = myList.toArray(test);
		double[] d = new double[myList.size()];
		for(int i = 0; i < myList.size(); i++)
		{
			d[i] = test[i];
		}
		return d;
	}
	
	public void reravel(double[] d)
	{
		for(int i = 0; i < W1.length; i++)
			for(int j = 0; j < W1[0].length; j++)
				W1[i][j] = d[i*W1[0].length+j];
		//System.out.println();

//		 printMatrix(W1);
		//System.out.println("Starting again...");
		for(int i = 0; i < W2.length; i++)
		{
			for(int j = 0; j < W2[0].length; j++)
			{
//				System.out.print(i + ", " + j + ": ");
//				System.out.println(i*W2[0].length+j+4 + ", ");
				W2[i][j] = d[i*W2[0].length+j+6];
			}
//			System.out.println("");
		}
		
//		 printMatrix(W2);
	}
	
	public void calculateForwardProp()
	{
		
		this.Z2 = this.calculateZ(this.X, this.W1, 2);
		this.A2 = Matrix.applySigmoid(this.Z2);
		
		this.Z3 = this.calculateZ(this.A2, this.W2, 3);
		
		this.yHat = Matrix.applySigmoid(this.Z3);
	}
	
	public void calculateCostFunctionPrimes()
	{
		if(true)
		{
			double epsilon = .00001;
	
			double[] unraveled = this.unravel(this.W1, this.W2);
			double[] perturbed = new double[unraveled.length];
			double[] gradient = new double[unraveled.length];
			for(int i = 0; i < perturbed.length; i++)
			{
				perturbed[i] = 0;
			}
			
			for(int i = 0; i < unraveled.length; i++)
			{
				unraveled[i] = unraveled[i] + epsilon;
				this.reravel(unraveled);
				calculateForwardProp();
				this.calculateCostFunction(this.Y, this.yHat);
				double pCost = this.cost;
				
				unraveled[i] = unraveled[i] - epsilon - epsilon;
				this.reravel(unraveled);
				calculateForwardProp();
				this.calculateCostFunction(this.Y, this.yHat);
				double nCost = this.cost;
				
				gradient[i] = (pCost - nCost) / (2*epsilon);
				
				unraveled[i] = unraveled[i] + epsilon;
				this.reravel(unraveled);
			}
			
			
			for(int i = 0; i < djdw1.length; i++)
				for(int j = 0; j < djdw1[0].length; j++)
					djdw1[i][j] = gradient[i*djdw1[0].length+j];
			System.out.println();
	
			
			System.out.println("Starting again...");
			for(int i = 0; i < djdw2.length; i++)
			{
				for(int j = 0; j < djdw2[0].length; j++)
				{
					djdw2[i][j] = gradient[i*djdw2[0].length+j+6];
				}
			}
		}
		else
		{
			double[][] D3temp = Matrix.subtract(this.yHat, this.Y);
			this.D3 = Matrix.multiplyVector(D3temp, Matrix.applySigmoidPrime(this.Z3));
			
			this.djdw2 = Matrix.multiply(Matrix.transpose(this.A2), this.D3);
	
			double[][] D2 = Matrix.multiplyScalar(Matrix.multiply(this.D3, Matrix.transpose(this.W2)), Matrix.applySigmoidPrime(this.Z2));
			this.djdw1 = Matrix.multiply(Matrix.transpose(this.X), D2);
		}
		

		if(false)
		{
			System.out.println("yhat");
			printMatrix(yHat);
			
			System.out.println("y");
			printMatrix(Y);

//			System.out.println("d3temp");
//			this.printMatrix(D3temp);

			System.out.println("sigmoidPrimeZ3");
			this.printMatrix(Matrix.applySigmoidPrime(this.Z3));

			System.out.println("delta3");
			this.printMatrix(this.D3);
			System.out.println();

			System.out.println("A2.T");
			this.printMatrix(Matrix.transpose(this.A2));
			System.out.println("sigmoidPrime(z3)");
			this.printMatrix(Matrix.transpose(Matrix.applySigmoidPrime(this.Z3)));
			
			System.out.println("delta2");
			this.printMatrix(D2);
			
		}
	}
	
	public double[][] calculateZ(double[][] X, double[][] W, int weightLevel)
	{
		double[][] forwardPropResults = Matrix.multiply(X, W);
		
		return forwardPropResults;
		
	}
	public double calculateCostFunction(double[][] Y, double[][] yHat)
	{
//		System.out.println("Calculating cost....");
		double[][] costResultMatrix = Matrix.subtract(Y, yHat);
		costResultMatrix = Matrix.multiplyVector(costResultMatrix, costResultMatrix);
		double costResult = Matrix.sumVector(costResultMatrix)/2;
		this.cost = costResult;
		
		return costResult;
	}
	
	public Matrix sigmoid(Matrix z)
	{
		
		return null;
	}
	
	public static void printMatrix(double[][] a)
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
	
	public NeuralNetwork()
	{
		inputLayerSize = 2;
		outputLayerSize = 1;
		hiddenLayerSize = 3;
	}

}
