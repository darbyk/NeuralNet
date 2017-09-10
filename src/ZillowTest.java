import com.darby.neuralnet.NeuralNet;


public class ZillowTest {
	
	public static void main(String args[])
	{
		Double[][] inputData = ImportAndSanitize.loadInputFile();
		Double[][] outputData = ImportAndSanitize.loadTrainFile();
		System.out.println("Data import done");
		
		int[] networkDescription = {58, 120, 60, 20, 5, 1};
		
		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "zillowTest5.txt", "zillowTest5.txt");
		System.out.println("Neural Network created");
		
		NN.loadFile();
		
		//Finally we can train our dataset
		NN.trainDataset();
		
		//Training now should be complete.  At this point we can save our weights, which is our actual computed answer.
		System.out.println("Saving Weights...");
		NN.saveWeights();
		
//		ImportAndSanitize.createTestOutput(NN);
		
	}
}
