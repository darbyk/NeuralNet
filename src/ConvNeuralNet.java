import java.util.ArrayList;

public class ConvNeuralNet {
	
	//Network Descriptors
	int[] networkDescription;
	int[] convNetworkDescription;
	ArrayList<Double[][][][]> networkFilters = new ArrayList<Double[][][][]>();
	
	
	public static void main(String[] args)
	{
		ConvNeuralNet cnn = new ConvNeuralNet();
		
		//Starting Unit Tests;
		//Represents my multiple images, across multiple color channels
		//This is a 4D array
		Double[][][][] myImage = {
			//Image 1 with 2 color channels
			{
				{
					{0.,0.,2.,1.,2.},
					{1.,2.,0.,2.,1.},
					{2.,2.,2.,1.,1.},
					{0.,2.,0.,1.,2.},
					{0.,2.,0.,0.,1.}
				},
				{
					{0.,0.,2.,1.,2.},
					{1.,2.,0.,2.,1.},
					{2.,2.,2.,1.,1.},
					{0.,2.,0.,1.,2.},
					{0.,2.,0.,0.,1.}
				}
			},
			//Image 2 with 2 color channels
			{
				{
					{0.,0.,2.,1.,2.},
					{1.,2.,0.,2.,1.},
					{2.,2.,2.,1.,1.},
					{0.,2.,0.,1.,2.},
					{0.,2.,0.,0.,1.}
				},
				{
					{0.,0.,2.,1.,2.},
					{1.,2.,0.,2.,1.},
					{2.,2.,2.,1.,1.},
					{0.,2.,0.,1.,2.},
					{0.,2.,0.,0.,1.}
				}
			}
		};
		
		//Represents my fitler set (I have multiple filter sets, where each filter set 
		//consists of multiple filters) I want to apply at the first convolution level
		Double[][][][] myFilter = {
			//Filter set 1 that works across 2 color channels
			{
					{
						{1.,1.,1.},
						{1.,1.,1.},
						{1.,1.,1.}
					},
					{
						{2.,2.,2.},
						{2.,2.,2.},
						{2.,2.,2.}
					}
			},
			//Filter set 2 that works across 2 color channels
			{
				{
					{1.,1.,1.},
					{1.,1.,1.},
					{1.,1.,1.}
				},
				{
					{2.,2.,2.},
					{2.,2.,2.},
					{2.,2.,2.}
				}
			}
		};
		
		//Want to create a CNN that has 2 convolution layers.  
		//Just adding the same filter again for now to see how this can be done
		//The depth of my networkFilters really determines the iterations I'll go.
		//Makes more sense to have a network description though.  
		//Lets break it into 2, one where we hold the CNN and one for the NN? Yes.
		cnn.networkFilters.add(myFilter);
		cnn.networkFilters.add(myFilter);
		
		//The stride we're applying over our image.  I'll need to add some validation maybe?
		//Because its important that I don't stride out of bounds 
		int stride = 1;
		
		//4D array is: number of images x number of filter sets x output length x output width
		int outputImageLength = (int) Math.ceil(myImage[0][0].length/(double)stride);
		int outputImageWidth = (int) Math.ceil(myImage[0][0][0].length/(double)stride);
		Double[][][][] result = new Double[myImage.length][cnn.networkFilters.get(0).length][outputImageLength][outputImageWidth];
		
		//Applies my filter sets across each image
		//But now I need to store the result in a 4D array
		for(int i = 0; i < myImage.length; i++)
		{
			for(int j = 0; j < cnn.networkFilters.get(0).length; j++)
			{
				result[i][j] = cnn.convolution(myImage[i], cnn.networkFilters.get(0)[j], stride);
			}
			
		}
		
		for(int i = 0; i < result.length; i++)
		{
			for(int j = 0; j < result[0].length; j++)
			{
				NeuralMatrix.printMatrix(result[i][j]);
				System.out.println("--New Matrix--");
			}
		}
	}
	
	public ConvNeuralNet()
	{
		
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

	
	private void initializeValuesInMatrix(boolean isRandom, ArrayList<Double[][]> initializeItems)
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
	
}
