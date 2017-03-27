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
	String saveFile = "test.txt";
	
	//Network Descriptors
	Double[][][][] inputData;
	Double[][] outputData;
	int[] networkDescription;
	int[][] convNetworkDescription;
	double complexityCostLambda = 0.0001;
	ArrayList<Double[][][][]> networkFilters = new ArrayList<Double[][][][]>();
	ArrayList<Double[][][][]> networkGradients = new ArrayList<Double[][][][]>();
	
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
			{3, 3, 2, 1},
			{3, 3, 1, 1},
			{3, 3, 1, 1},
			{3, 3, 2, 1}
		};
		//First parameter is calculated at another time
		int[] netDesc = {
			0, 5, 1	
		};
		ConvNeuralNet cnn = new ConvNeuralNet(netDesc, cnnNetDesc);
		
		System.out.println(cnn.numberOfVariables);
		
		
		//Starting Unit Tests;
		//Represents my multiple images, across multiple color channels
		//This is a 4D array
		//myImage[x][][][] = number of input images
		//myImage[][x][][] = number of input channels (depth dimension of your image)
		//myImage[][][x][] = length dimension of your image
		//myimage[][][][x] = width dimension of your image
		Double[][][][] myImages = {
			//Image 1 with 3 color channels
			{
				{
					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
				},
				{
					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
				},
				{
					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
				}
			},
			//Image 2 with 3 color channels
			{
				{
					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
				},
				{
					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
				},
				{
					{0.,0.,2.,1.,2.,1.,2.,3.,0.},
					{1.,2.,0.,2.,1.,0.,3.,2.,1.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{0.,2.,0.,1.,2.,0.,1.,2.,0.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.},
					{2.,2.,2.,1.,1.,0.,0.,1.,2.},
					{0.,2.,0.,0.,1.,0.,2.,1.,2.}
				}
			}
		};
		
		cnn.inputData = myImages;
		
		//Represents my fitler set (I have multiple filter sets, where each filter set 
		//consists of multiple filters) I want to apply at the first convolution level
		//myFilter[x][][][] = the number of filter sets I have at one layer in my CNN
		//myFilter[][x][][] = the depth of the filter set at a particular layer
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
		
		cnn.calculateForwardProp();
		
		
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
		diag = new double [ numberOfVariables ];
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
				for(int k = 0; k < networkFilters.get(0).length; k++)
				{
					//cnn.convolution returns a 2D matrix, hence we store it in our result
					z[j][k] = convolution(a[j], networkFilters.get(0)[k], stride);
				}	
			}
			a = NeuralMatrix.applyRelu(z);
		}
		yHatCNN = a;
		
		//Print out some random reuslt
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("Printing out the final image image convolution result");
		
		for(int i = 0; i < yHatCNN.length; i++)
		{
			for(int j = 0; j < yHatCNN[0].length; j++)
			{
				NeuralMatrix.printMatrix(yHatCNN[i][j]);
				System.out.println("--New Matrix--");
			}
		}
		
		Double[][] nnInput = transformVolumeToNNInput();
		NeuralMatrix.printMatrix(nnInput);
		
		Double[][] nnOutput = {
			{1.}, 
			{1.}
		};
		
		networkDescription[0] = nnInput[0].length;
		
		NeuralNet nn = new NeuralNet(networkDescription, nnInput, nnOutput);
		nn.calculateForwardProp();
		yHat = nn.yHat;
		NeuralMatrix.printMatrix(yHat);
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
	public double[] unravel(ArrayList<Double[][][][]> unravelItems)
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
	public void reravel(double[] dArray, ArrayList<Double[][][][]> reravelItem)
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
	}
	
}
