
public class ConvNeuralNet {
	
	
	
	public static void main(String[] args)
	{
		ConvNeuralNet cnn = new ConvNeuralNet();
		
		//Starting Unit Tests;
		Double[][] myImage = {
				{0.,0.,2.,1.,2.},
				{1.,2.,0.,2.,1.},
				{2.,2.,2.,1.,1.},
				{0.,2.,0.,1.,2.},
				{0.,2.,0.,0.,1.}
		};
		Double[][] myFilter = {
				{1.,1.,1.},
				{1.,1.,1.},
				{1.,1.,1.}
		};
		Double[][] result = cnn.convolution(myImage, myFilter);
		NeuralNet.printMatrix(result);
	}
	
	public ConvNeuralNet()
	{
		
	}
	
	public Double[][] convolution(Double[][]image, Double[][] filter)
	{
		int imageLength = image.length;
		int imageWidth = image[0].length;
		
		Double[][] convolvedImage = new Double[imageLength][imageWidth];
		
		for(int i = 0; i < imageLength; i++)
		{
			for(int j = 0; j < imageWidth; j++)
			{
				convolvedImage[i][j] = pointConvolution(image, filter, i, j);
			}
			
		}
		return convolvedImage;
	}
	
	public double pointConvolution(Double[][] image, Double[][] filter, int imageX, int imageY)
	{
		int filterLength = filter.length;
		int filterWidth = filter[0].length;
		int imageLength = image.length;
		int imageWidth = image[0].length;
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
				double imageValue = 0.0;
				if(imageIndexX >= 0 && imageIndexY >= 0 && imageIndexX < imageLength && imageIndexY < imageWidth)
				{
					imageValue = image[imageIndexX][imageIndexY];
				}
				
				//Dot product
				resultPoint += filter[i][j] * imageValue;
				filterSummation += filter[i][j];
			}
		}
//		resultPoint = resultPoint / filterSummation;
		
		return resultPoint;
	}
	
}
