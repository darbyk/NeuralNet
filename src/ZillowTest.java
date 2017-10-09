import com.darby.neuralnet.NeuralNet;


public class ZillowTest {
	
	public static void main(String args[])
	{
		Double[][] inputData = ImportAndSanitize.loadInputFile("massagedPropertyTrain.csv");
		Double[][] outputData = ImportAndSanitize.loadTrainFile("massagedTrain.csv");
		
//		Double[][] inputData = ImportAndSanitize.loadInputFile("darbyMapping.csv");
//		Double[][] outputData = ImportAndSanitize.loadTrainFile("darbyMappingOutput.csv");
		System.out.println("Data import done");
		
		int[] networkDescription = {59, 110, 80, 25, 1};
		
		NeuralNet NN = new NeuralNet(networkDescription, inputData, outputData, "zillowTest4.txt", "zillowTest4.txt");
		System.out.println("Neural Network created");
		
		NN.loadFile();
		
		//Finally we can train our dataset
//		NN.trainDataset();
		
		//Training now should be complete.  At this point we can save our weights, which is our actual computed answer.
		System.out.println("Saving Weights...");
//		NN.saveWeights();
		
		ImportAndSanitize.createTestOutput(NN);
		
	}
}
