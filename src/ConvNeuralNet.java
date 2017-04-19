//Notes: 
//1. Something is probably wrong with the normalization method of the neural network.
//    We probably don't want to normalize each time for an image.  We'll likely have 0
//    values in our input due to the ReLu function, so it might be normalized on time,
//    but not another time.  Or, each time we change the input as well, it might
//    mess up the inputs if they aren't normalized either...It goes against the basic
//    neural net architecture, but I think we may want to hide it or normalize it to 
//    255 always (largest expected pixel value)
//2. Thoughts?

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class ConvNeuralNet {
	//LBFGS variables
	int iflag[] = {0};
	int iprint[] = new int[2];
	int icall;
	double diag[];
	int numberOfVariables=0;
	String filePath = "C:\\NeuralNet\\";
	String loadFile = "test.txt";
	String saveFile = "test2.txt";
	
	//Network Descriptors
	Double[][][][] inputData;
	Double[][] outputData;
	int[] networkDescription;
	int[][] convNetworkDescription;
	double complexityCostLambda = 0.0001;
	ArrayList<Double[][][][]> networkFilters = new ArrayList<Double[][][][]>();
	ArrayList<Double[][][][]> networkGradients = new ArrayList<Double[][][][]>();
	
	//Neural Net
	NeuralNet nn;
	
	//Results
	Double[][] yHat;
	Double[][][][] yHatCNN;
	double cost;
	
	
	public static void main(String[] args)
	{
		//Begin CNN initialization tests
		//convNetworkDescription[x][0] = filterSize (dimensions)
		//convNetworkDescription[x][1] = number of filters
		//convNetworkDescription[x][2] = stride
		//convNetworkDescription[x][3] = padding
		int[][] cnnNetDesc = {
			{5, 3, 2, 1},
			{3, 3, 1, 1},
			{3, 3, 1, 1},
			{3, 3, 2, 1},
			{3, 3, 1, 1},
			{3, 3, 2, 1}
		};
		//First parameter is calculated at another time
		int[] netDesc = {
			48, 30, 1
		};
		
		
		//Starting Unit Tests;
		//Represents my multiple images, across multiple color channels
		//This is a 4D array
		//myImage[x][][][] = number of input images
		//myImage[][x][][] = number of input channels (depth dimension of your image)
		//myImage[][][x][] = length dimension of your image
		//myimage[][][][x] = width dimension of your image
//		Double[][][][] myImages = {
//			//Image 1 with 3 color channels
//			{
//				{
//					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
//					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
//				},
//				{
//					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
//					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
//				},
//				{
//					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
//					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
//				}
//			},
//			//Image 2 with 3 color channels
//			{
//				{
//					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
//					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
//				},
//				{
//					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
//					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
//				},
//				{
//					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
//					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
//					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
//					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
//				}
//			}
//		};
		
		ImageHelper helper = new ImageHelper("data\\");
		
		Double[][][][] myImages = {
				helper.extractBytesFullChannels("eight1.png"),
				helper.extractBytesFullChannels("eight2.png"),
				helper.extractBytesFullChannels("one1.png")
		};
		
		Double[][] outputData = {
				{1.},
				{1.},
				{0.}
		};
		
		ConvNeuralNet cnn = new ConvNeuralNet(netDesc, cnnNetDesc, myImages, outputData);
		
		System.out.println(cnn.numberOfVariables);
		
		
//		cnn.inputData = myImages;
		
		/*
		ArrayList<Double[][][][]> temp = new ArrayList<Double[][][][]>();
		for(int a = 0; a  < 2; a ++)
		{
			Double[][][][] layer = new Double[2][2][2][2];
			for(int b = 0; b < 2; b++)
			{
				for(int c = 0; c < 2; c++)
				{
					for(int d = 0; d < 2; d++)
					{
						for(int e = 0; e < 2; e++)
						{
							layer[b][c][d][e] = e + a*16 + b*8 + c*4 + d*2 + 0.0;
						}
					}
				}
			}
			temp.add(layer);
		}
		
		
		ArrayList<Double[][]> tempSmall = new ArrayList<Double[][]>();
		for(int a = 0; a  < 2; a ++)
		{
			Double[][] layer = new Double[2][2];
			for(int b = 0; b < 2; b++)
			{
				for(int c = 0; c < 2; c++)
				{
					layer[b][c] = a*4 + b*2 + c + 0.0;
				}
			}
			tempSmall.add(layer);
		}
		
		System.out.println("Testing");
		
		double[] unravelTest = cnn.unravel(temp, tempSmall);
		
		unravelTest[30] = unravelTest[30] + 1;
		unravelTest[35] = unravelTest[35] + 1;
		
		cnn.reravel(unravelTest, temp, tempSmall);
		
		System.out.println("Testing");
		
		*/
		
		//Represents my fitler set (I have multiple filter sets, where each filter set 
		//consists of multiple filters) I want to apply at the first convolution level
		//myFilter[x][][][] = the number of filter sets I have at one layer in my CNN
		//myFilter[][x][][] = the depth of the filter set at a particular layer
		//						the depth of a filter set corresponds to the number of filters at the previous layer
		//myFilter[][][x][] = the length of a filter
		//myFilter[][][][x] = the widht of a filter
//		Double[][][][] myFilter = {
//			//Filter set 1 that works across 2 color channels
//			{
//					{
//						{1.,1.,1.},
//						{1.,1.,1.},
//						{1.,1.,1.}
//					},
//					{
//						{2.,2.,2.},
//						{2.,2.,2.},
//						{2.,2.,2.}
//					}
//			},
//			//Filter set 2 that works across 2 color channels
//			{
//				{
//					{1.,1.,1.},
//					{1.,1.,1.},
//					{1.,1.,1.}
//				},
//				{
//					{2.,2.,2.},
//					{2.,2.,2.},
//					{2.,2.,2.}
//				}
//			}
//		};
		
		//Want to create a CNN that has 2 convolution layers.  
		//Just adding the same filter again for now to see how this can be done
		//The depth of my networkFilters really determines the iterations I'll go.
		//Makes more sense to have a network description though.  
		//Lets break it into 2, one where we hold the CNN and one for the NN? Yes.
//		cnn.networkFilters.add(myFilter);
//		cnn.networkFilters.add(myFilter);
		
		//The stride we're applying over our image.  I'll need to add some validation maybe?
		//Because its important that I don't stride out of bounds 
		int stride = 1;
		
		//4D array is: number of images x number of filter sets x output length x output width
		//result[x][][][] = the result of the convolution of each image, 
		//       so result[1][][][] holds the convolution result for image 1 - this mimicks bulk processing
		//result[][x][][] = the depth dimension of my convolution volume output for a given image
		//result[][][x][] = the length dimension of my convolution volume output
		//result[][][][x] = the width dimension of my convolution volume output
		int outputImageLength = (int) Math.ceil(myImages[0][0].length/(double)stride);
		int outputImageWidth = (int) Math.ceil(myImages[0][0][0].length/(double)stride);
		Double[][][][] result = new Double[myImages.length][cnn.networkFilters.get(0).length][outputImageLength][outputImageWidth];
		
		//Applies my filter sets across each image
		//But now I need to store the result in a 4D array
		//Repeat this method over and over again (this is kinda my forward prop
//		for(int i = 0; i < myImages.length; i++)
//		{
//			for(int j = 0; j < cnn.networkFilters.get(0).length; j++)
//			{
//				//cnn.convolution returns a 2D matrix, hence we store it in our result
//				result[i][j] = cnn.convolution(myImages[i], cnn.networkFilters.get(0)[j], stride);
//			}
//			
//		}
		
		
		
		cnn.saveWeights();
		
		
		
		
		
		/*
		cnn.calculateForwardProp();
		
		
		cnn.calculateCostFunctionPrimes();
		System.out.println("******************");
		System.out.println("Calculating Primes");		
		
		for(int i = 0; i < cnn.networkGradients.size(); i++)
		{
			for(int j = 0; j < cnn.networkGradients.get(i).length; j++)
			{
				for(int k = 0; k < cnn.networkGradients.get(i)[j].length; k++)
				{
					NeuralMatrix.printMatrix(cnn.networkGradients.get(i)[j][k]);
					System.out.println("--New Matrix--");
				}
			}
			System.out.println("*");
		}
		
		System.out.println("Printing NN Gradients");
		for(int i = 0; i < cnn.nn.networkDescription.length-1; i++)
		{
			NeuralMatrix.printMatrix(cnn.nn.gradients.get(i));
			System.out.println("*");
		}
		
		
		//Save some temp variables and begin unraveling the weights for back propagation
		double cost2 = cnn.cost;
		double[] unraveledWeights = cnn.unravel(cnn.networkFilters, cnn.nn.weights);
		double[] unraveledGradient = cnn.unravel(cnn.networkGradients, cnn.nn.gradients);
		System.out.println("Beginning Back propogation...");
		try {

			do
			{
				LBFGS.lbfgs((cnn.numberOfVariables + cnn.nn.numberOfVariables), 300, unraveledWeights, cost2, unraveledGradient, false, cnn.diag, cnn.iprint, 1.0e-4, 1.0e-17, cnn.iflag);
				cnn.reravel(unraveledWeights, cnn.networkFilters, cnn.nn.weights);
				cnn.calculateForwardProp();
				cnn.calculateCostFunctionPrimes();
				cnn.calculateCostFunction(cnn.outputData, cnn.yHat);
				cost2 = cnn.cost;
				unraveledGradient = cnn.unravel(cnn.networkGradients, cnn.nn.gradients);
				NeuralMatrix.printMatrix(cnn.yHat);
		
			} while(cnn.iflag[0] != 0);

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("error");
		}
		*/
		//Back propagation complete. 
		
		
		
	}
	
	public ConvNeuralNet()
	{
		iprint [ 1 -1] = 1;
		iprint [ 2 -1] = 0;
		iprint [ 0] = 1;

		iflag[0] = 0;
		icall = 0;
	}
	public ConvNeuralNet(int[] networkDescription, int[][] convNetworkDescription)
	{
		//Setup for LBFGS
		this();
		
		//Initialize Network Descriptions
		this.networkDescription = networkDescription;
		this.convNetworkDescription = convNetworkDescription;
		
		//Initialize Weights
		initializeValuesInMatrix5D(true, networkFilters);
		
		//Initialize Gradients
		initializeValuesInMatrix5D(false, networkGradients);
		
		for(int i = 0; i < convNetworkDescription.length; i++)
		{
			numberOfVariables += networkFilters.get(i).length * 
					networkFilters.get(i)[0].length *
					networkFilters.get(i)[0][0].length *
					networkFilters.get(i)[0][0][0].length;
//			numberOfVariables += (convNetworkDescription[i][1] * convNetworkDescription[i][0] * convNetworkDescription[i][0]);
		}
		
		nn = new NeuralNet(networkDescription);
		
		diag = new double [ numberOfVariables + nn.numberOfVariables];
	}
	public ConvNeuralNet(int[] networkDescription, int[][] convNetworkDescription, Double[][][][] inputData, Double[][] outputData)
	{
		this(networkDescription, convNetworkDescription);
		this.inputData = inputData;
		this.outputData = outputData;
		
		File weightInitiatorPath = new File(filePath + loadFile);
		
		try(BufferedReader br = new BufferedReader(new FileReader(weightInitiatorPath))) {
			StringBuilder sb = new StringBuilder();
			
			String line;
			for(int i = 0; i < convNetworkDescription.length; i++)
			{
				line = br.readLine();
				String[] splitter = line.split(" ");
			    networkDescription = new int[splitter.length];
			    for(int j = 0; j < splitter.length; j++)
			    {
			    	convNetworkDescription[i][j] = Integer.parseInt(splitter[j]);
			    }
				
			}
			
		    
		    line = br.readLine();
		    line = br.readLine();
		    int counter = 0;
		    while (!line.equals("EOF") && line != null) {
		    	//First need to get the number of filter sets at this layer
		    	int numOfFilterSets = convNetworkDescription[counter][1];
		    	
		    	//Next get the depth of the filter sets (3 at layer 0, or previous layers filters
		    	int depthOfFilterSets = 3;
		    	if(counter != 0)
		    	{
		    		depthOfFilterSets = convNetworkDescription[counter-1][1];
		    	}
		    	
		    	//Finally get the dimension of your filter
		    	int filterDimension = convNetworkDescription[counter][0];
		    	
		    	//Construct this layer's temp weights
		    	Double[][][][] tempWeights = new Double[numOfFilterSets][depthOfFilterSets][filterDimension][filterDimension];


		    	//Need to do something complicated
		    	int numberOfFilterSetsReached = 0;
		    	int numberOfFiltersInSetReached = 0;
		    	int indexOfFilterLength = 0;
		    	while (
		    			!line.equals("EOF") && 
		    			(!line.equals("----") && numberOfFilterSetsReached != numOfFilterSets  && numberOfFiltersInSetReached != depthOfFilterSets) 
		    			&& line != null) 
		    	{
		    		if(line == "----")
		    		{
		    			if(numberOfFiltersInSetReached == depthOfFilterSets)
		    			{
		    				numberOfFiltersInSetReached = 0;
		    				indexOfFilterLength = 0;
		    				numberOfFilterSetsReached++;
		    				counter++;
		    			}
		    			else
		    			{
		    				indexOfFilterLength = 0;
		    				numberOfFiltersInSetReached++;
		    			}
		    		}
		    		else
		    		{
		    			String[] matrixSplitter = line.split(" ");
					    
					    for(int i = 0; i < matrixSplitter.length; i++)
					    {
					    	tempWeights[numberOfFiltersInSetReached][numberOfFilterSetsReached][indexOfFilterLength][i] = Double.parseDouble(matrixSplitter[i]);
					    }
					    
					    indexOfFilterLength++;				        
		    		}
		    		
		    		
		    		line = br.readLine();
			        if(!line.equals("----") && numberOfFilterSetsReached != numOfFilterSets  && numberOfFiltersInSetReached != depthOfFilterSets)
			        {
			        	networkFilters.add(tempWeights);
			        }

		    	}
//		    	counter++;
		        line = br.readLine();
		    }
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
	
	public void calculateForwardProp()
	{
		Double[][][][] a = inputData;
		for(int i = 0; i < convNetworkDescription.length; i++)
		{
			int stride = convNetworkDescription[i][2];
			int outputImageLength = (int) Math.ceil(a[0][0].length/(double)stride);
			int outputImageWidth = (int) Math.ceil(a[0][0][0].length/(double)stride);
			Double[][][][] z = new Double[a.length][networkFilters.get(0).length][outputImageLength][outputImageWidth];
			
			//z = a.convolve(filter)
			for(int j = 0; j < a.length; j++)
			{
				//This was 0/i too....Has to be i not 0
				for(int k = 0; k < networkFilters.get(i).length; k++)
				{
					//cnn.convolution returns a 2D matrix, hence we store it in our result
					//Oops...Is it i or 0 here....It has to be i, I think...if its 0, I think we 
					//apply the same initial filter over the entire set if it is 0, and therefore, we want
					//to make sure it is .get(i) to make sure we use every variable...but why 
					//didn't it break everywhere else, either one solves the problem
					z[j][k] = convolution(a[j], networkFilters.get(i)[k], stride);
				}	
			}
			a = NeuralMatrix.applyRelu(z);
		}
		yHatCNN = a;
		
		//Print out some random result
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("Printing out the final image convolution result");
		
		for(int i = 0; i < yHatCNN.length; i++)
		{
			for(int j = 0; j < yHatCNN[0].length; j++)
			{
				NeuralMatrix.printMatrix(yHatCNN[i][j]);
				System.out.println("--New Matrix--");
			}
		}
		
		System.out.println();
		System.out.println("Printing out the transformed Neural Net input");
		Double[][] nnInput = transformVolumeToNNInput();
		NeuralMatrix.printMatrix(nnInput);
				
//		networkDescription[0] = nnInput[0].length;
//		NeuralNet nn = new NeuralNet(networkDescription, nnInput, outputData);
		nn.inputData = nnInput;
		nn.outputData = outputData;
		nn.normalizeMatrix(nn.inputData);
		
		System.out.println("Beginning neural net");
		nn.calculateForwardProp();
		yHat = nn.yHat;
		
		System.out.println("");
		System.out.println("NN Output");
		NeuralMatrix.printMatrix(yHat);
		
		
		cost = calculateCostFunction(outputData, yHat);
		System.out.println("Cost output is");
		
		System.out.println(cost);
		
		
		
	}
	
	//Some method to transform the output of the forwardpropogation of CNN to input of NN
	public Double[][] transformVolumeToNNInput()
	{
		//This is really just unraveling my matrix into a 2D matrix
		
		int numberOfNodes = yHatCNN[0].length *
				yHatCNN[0][0].length *
				yHatCNN[0][0][0].length;
		
		
		//yHatCNN.length is the number of equations
		Double[][] nnInputs = new Double[yHatCNN.length][numberOfNodes];
		for(int i = 0; i < yHatCNN.length; i++)
		{
			nnInputs[i] = unravel(yHatCNN[i]);
		}
		
		return nnInputs;
	}
	
	public Double[][] convolution(Double[][][] image, Double[][][] filter, int stride)
	{
		int imageDepth = image.length;
		int imageLength = image[0].length;
		int imageWidth = image[0][0].length;
		
		
		Double[][] convolvedImage = new Double[(int) Math.ceil(imageLength/(double)stride)][(int) Math.ceil(imageWidth/(double)stride)];
		
		for(int i = 0; i < imageLength; i=i+stride)
		{
			for(int j = 0; j < imageWidth; j=j+stride)
			{
				convolvedImage[i/stride][j/stride] = pointConvolution(image, filter, i, j);
			}
			
		}
		return convolvedImage;
	}
	
	private double pointConvolution(Double[][][] image, Double[][][] filter, int imageX, int imageY)
	{
		int filterDepth = filter.length;
		int filterLength = filter[0].length;
		int filterWidth = filter[0][0].length;
		int imageLength = image[0].length;
		int imageWidth = image[0][0].length;
		int filterOffset = filterLength / 2;
		
		double resultPoint = 0.0;
		double filterSummation = 0.0;
		
		for(int i = 0; i < filterLength; i++)
		{
			for(int j = 0; j < filterWidth; j++)
			{
				int imageIndexX = i - filterOffset + imageX;
				int imageIndexY = j - filterOffset + imageY;
				
				//Effectively Padding image with 0
				//And also initialize an image depth slice
				double[] imageValue = new double[filterDepth];
				for(int k = 0; k < filterDepth; k++)
				{
					imageValue[k] = 0.0;
				}
				
				if(imageIndexX >= 0 && imageIndexY >= 0 && imageIndexX < imageLength && imageIndexY < imageWidth)
				{
					for(int k = 0; k < filterDepth; k++)
					{
						imageValue[k] = image[k][imageIndexX][imageIndexY];
					}
				}
				
				//Dot product
				for(int k = 0; k < filterDepth; k++)
				{
					resultPoint += filter[k][i][j] * imageValue[k];
					filterSummation += filter[k][i][j];
				}
				
			}
		}
//		resultPoint = resultPoint / filterSummation;
		
		return resultPoint;
	}

	
	private void initializeValuesInMatrix3D(boolean isRandom, ArrayList<Double[][]> initializeItems)
	{
		System.out.println("Initializing Weights...");
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
			NeuralMatrix.printMatrix(tempLayer);
		}
	}
	
	//convNetworkDescription[x][0] = filterSize (dimensions)
	//convNetworkDescription[x][1] = number of filters
	//convNetworkDescription[x][2] = stride
	//convNetworkDescription[x][3] = padding
	//
	//filterSet[][][][] create an array of filterSets
	//filterSet[x][][][] = the number of filter sets at a given location the CNN
	//filterSet[][x][][] = the number of filters needed to properly create a filter set (depth dimension)
	//filterSet[][][x][] = the length dimension of the filter set
	//filterSet[][][][x] = the width dimension of the filter set
	private void initializeValuesInMatrix5D(boolean isRandom, ArrayList<Double[][][][]> initializeItems)
	{
		System.out.println("Initializing Weights...");
		
		//Add some random stuff in
		for(int i = 0; i < convNetworkDescription.length; i++)
		{
			Double tempLayer[][][][];
			if(i == 0)
			{
				tempLayer = new Double
						[convNetworkDescription[0][1]]
						[3] //we have an image with 3 color channels coming in
						[convNetworkDescription[0][0]]
						[convNetworkDescription[0][0]];
			}
			else
			{
				tempLayer = new Double
						[convNetworkDescription[i][1]] //number of filters to apply at this region
						[convNetworkDescription[i-1][1]] //number of filters in the filter set -> This must be equal to the number of filters in the previous layer
						[convNetworkDescription[i][0]] //length of a filter
						[convNetworkDescription[i][0]]; //width of a filter	
			}
			
			for(int x = 0; x < tempLayer.length; x++)
			{
				for(int y = 0; y < tempLayer[0].length; y++)
				{
					for(int z = 0; z < tempLayer[0][0].length; z++)
					{
						for(int a = 0; a < tempLayer[0][0][0].length; a++)
						{
							if(isRandom)
							{
								double randomNum = Math.random()*2-1;
								int curWeightInt = ((int)(randomNum * 1000));
								double curWeight = curWeightInt/1000.0;
								tempLayer[x][y][z][a] = curWeight;
							}
							else
							{
								tempLayer[x][y][z][a] = 0.0;
							}
						}
						
					}
					
				}
			}
			initializeItems.add(tempLayer);
			NeuralMatrix.printMatrix(tempLayer);
		}
	}
	
	
	//Helper method.  Takes in some 5D matrix and converts into a 1D matrix
	public double[] unravel(ArrayList<Double[][][][]> unravelItems, ArrayList<Double[][]> unravelitemsNeuralNet)
	{
		ArrayList<Double> myList = new ArrayList<Double>();
		for(int i = 0; i < unravelItems.size(); i++)
		{
			Double[][][][] curWeightLayer = unravelItems.get(i);
			for(int a = 0; a < curWeightLayer.length; a++)
			{
				for(int b = 0; b < curWeightLayer[0].length; b++)
				{
					for(int c = 0; c < curWeightLayer[0][0].length; c++)
					{
						for(int d = 0; d < curWeightLayer[0][0][0].length; d++)
						{
							myList.add(curWeightLayer[a][b][c][d]);
						}
					}
				}
			}
		}
		
		for(int i = 0; i < unravelitemsNeuralNet.size(); i++)
		{
			Double[][] curWeightLayer = unravelitemsNeuralNet.get(i);
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

	//Helper method.  Takes in some 3D matrix and converts into a 1D matrix
	public Double[] unravel(Double[][][] unravelItems)
	{
		ArrayList<Double> myList = new ArrayList<Double>();
		for(int i = 0; i < unravelItems.length; i++)
		{
			Double[][] curWeightLayer = unravelItems[i];
			for(int a = 0; a < curWeightLayer.length; a++)
			{
				for(int b = 0; b < curWeightLayer[0].length; b++)
				{
					myList.add(curWeightLayer[a][b]);
				}
			}
		}
		
		Double[] unraveledList = new Double[myList.size()];
		unraveledList = myList.toArray(unraveledList);
		
		return unraveledList;
	}
	
	
	//Helper method.  Takes in a 1D matrix and converts it into the proper 5D matrix
	public void reravel(double[] dArray, ArrayList<Double[][][][]> reravelItem, ArrayList<Double[][]> reravelItemNeuralNet)
	{
		int dPosition = 0;
		for(int i = 0; i < reravelItem.size(); i++)
		{
			Double[][][][] curWeightLayer = reravelItem.get(i);
			for(int a = 0; a < curWeightLayer.length; a++)
			{
				for(int b = 0; b < curWeightLayer[0].length; b++)
				{
					for(int c = 0; c < curWeightLayer[0][0].length; c++)
					{
						for(int d = 0; d < curWeightLayer[0][0][0].length; d++)
						{
							curWeightLayer[a][b][c][d] = dArray[dPosition];
							dPosition++;
						}
					}
				}
			}
			reravelItem.set(i, curWeightLayer);
		}
		
		for(int i = 0; i < reravelItemNeuralNet.size(); i++)
		{
			Double[][] curWeightLayer = reravelItemNeuralNet.get(i);
			for(int x = 0; x < curWeightLayer.length; x++)
			{
				for(int y = 0; y < curWeightLayer[0].length; y++)
				{
					curWeightLayer[x][y] = dArray[dPosition];
					dPosition++;
				}
			}
			reravelItemNeuralNet.set(i, curWeightLayer);
		}
		
		
	}
	
	//Calculates how wrong our current state is
	public double calculateCostFunction(Double[][] Y, Double[][] yHat)
	{
		//Calculate weight complexity
		//double complexityCost = this.calculateWeightComplexity();
		
		//Calculate the yHat - Y, square it, then half it (1/2) (y-yHat)^2
		Double[][] costResultMatrix = NeuralMatrix.subtract(Y, yHat);
		costResultMatrix = NeuralMatrix.multiplyVector(costResultMatrix, costResultMatrix);
		double answerCost = NeuralMatrix.sumVector(costResultMatrix)/2;
		
		//Sum our MatrixCost and our weightCost
		//this.cost = answerCost + complexityCost;
		this.cost = answerCost;
		
		return this.cost;
	}
	
	
	public void calculateCostFunctionPrimes()
	{
		double epsilon = .00000001;

		double[] unraveled = this.unravel(networkFilters, nn.weights);
		double[] gradient = new double[unraveled.length];
		
		//We come into this method with varying weights from the original set sometimes.  Therefore, calculate the cost and store it.
		calculateForwardProp();
		this.calculateCostFunction(outputData, yHat);
		double origCost = this.cost;

		System.out.println("unraveled length: " + unraveled.length);
		for(int i = 0; i < unraveled.length; i++)
		{
			unraveled[i] = unraveled[i] + epsilon;
			this.reravel(unraveled, this.networkFilters, nn.weights);
			calculateForwardProp();
			this.calculateCostFunction(outputData, yHat);
			double pCost = this.cost;

			gradient[i] = (pCost - origCost) / (epsilon);

			unraveled[i] = unraveled[i] - epsilon;
			this.reravel(unraveled, this.networkFilters, nn.weights);
		}
		this.reravel(gradient, this.networkGradients, nn.gradients);
	}
	
	//Save the weights to the proper save file
	public void saveWeights()
	{
		try(  PrintWriter out = new PrintWriter( this.filePath + this.saveFile )  ){
			ArrayList<String> convNetworkDescriptionString = new ArrayList<String>();
			
			for(int i = 0; i < convNetworkDescription.length; i++)
			{
				String convNetworkLayer = "";
				for(int j = 0; j < convNetworkDescription[0].length; j++)
				{
					convNetworkLayer += convNetworkDescription[i][j] + " ";
				}
				convNetworkDescriptionString.add(convNetworkLayer);
			}
			
			ArrayList<String> weightsString = new ArrayList<String>();
			for(int i = 0; i < networkFilters.size(); i++)
			{
				Double[][][][] d = networkFilters.get(i);
				for(int a = 0; a < d.length; a++)
				{
					for(int b = 0; b < d[a].length; b++)
					{
						for(int c = 0; c < d[a][b].length; c++)
						{
							Double[] dx = d[a][b][c];
							String curWeightLine = "";
							for(int y = 0; y < dx.length; y++)
							{
								curWeightLine += dx[y] + " ";
							}
							weightsString.add(curWeightLine);
						}
						weightsString.add("----");
					}
				}
			}
			
			for(String s : convNetworkDescriptionString)
			{
				out.println(s);
			}
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

	
}
