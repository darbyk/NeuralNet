import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/* Nothing more than a helper file */

public class Mapping {

	public static void main(String args[])
	{
//		HashSet<Double> resultsWithAnswers = readOutputFile();
//		readInputFile(resultsWithAnswers);
//		
//		
//		reorderOutput();
		
		
//		HashSet<Double> SampleSubmissionAnswers = readSampleSolutionFile();
//		readPropertiesForOutputFile(SampleSubmissionAnswers);
		
		findParcelId("13026021");
		
	}
	
	
	
	public static void findParcelId(String parcelId)
	{
		
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\properties_2016.csv");
		
		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
//			PrintWriter out = new PrintWriter( "C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbyMapping.csv" );
			
			String line = br.readLine();
			line = br.readLine();
		    int counter = 0;
		    boolean matchesFound = false;
			
		    
		    while (!matchesFound && line != null ) {
		    	
		    	Double[] tempLine = new Double[57];
		    	String[] splitLine = line.split(",");
		    	
//		    	System.out.println(splitLine[0]);
		    	
		    	if(parcelId.compareTo(splitLine[0]) == 0)
		    	{
		    		System.out.println(line);
		    		matchesFound = true;
		    	}
		    		
		    	line = br.readLine();
		    	if(counter % 100000 == 0)
		    		System.out.println("Counter at: " + counter);
		    	counter++;
		    }
		    
		    System.out.println(matchesFound);
		    
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	
	public static HashSet<Double> readOutputFile()
	{
		HashSet<Double> hs = new HashSet<Double>();
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\train_2016_v2.csv\\train_2016_v2.csv");
		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			String line = br.readLine();
			line = br.readLine();
		    
		    while (line != null) {
		    	
		    	Double[] tempLine = new Double[2];
		    	String[] splitLine = line.split(",");
		    	Double tempValueBeforeSanitation = null;
		    	tempValueBeforeSanitation = Double.parseDouble(splitLine[0]);
		    	hs.add(tempValueBeforeSanitation);
		    	line = br.readLine();
		    }
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println(hs.size());
		return hs;
	}
	
	
	public static void readInputFile(HashSet<Double> resultsToMatch)
	{
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\properties_2016.csv");

		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			PrintWriter out = new PrintWriter( "C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbyMapping.csv" );
			
			String line = br.readLine();
			line = br.readLine();
		    int counter = 0;
		    int matchesFound = 0;
			
		    while (line != null ) {
		    	
		    	Double[] tempLine = new Double[57];
		    	String[] splitLine = line.split(",");
		    	
		    	Double valueToMatch = Double.parseDouble(splitLine[0]);
		    	
		    	if(resultsToMatch.contains(valueToMatch))
		    	{
//		    		System.out.println("Found match: " + splitLine[0]);
		    		matchesFound++;
		    		out.println(line);
		    	}
		    		
		    	line = br.readLine();
		    	if(counter % 100000 == 0)
		    		System.out.println("Counter at: " + counter);
		    	counter++;
		    }
		    
		    System.out.println(matchesFound);
		    
		    br.close();
		    out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	

	public static void reorderOutput()
	{
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbyMapping.csv");
		File outputValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\train_2016_v2.csv\\train_2016_v2.csv");

		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			PrintWriter out = new PrintWriter( "C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbyMappingOutput.csv" );
			
			String line = br.readLine();
			line = br.readLine();
		    int counter = 0;
		    int matchesFound = 0;
			
		    while (line != null ) {
		    	
		    	Double[] tempLine = new Double[57];
		    	String[] splitLine = line.split(",");
		    	
		    	Double valueToMatch = Double.parseDouble(splitLine[0]);
		    	
		    	try(BufferedReader outputBr = new BufferedReader(new FileReader(outputValuesPath))){
		    		
					String outputLine = outputBr.readLine();
					outputLine = outputBr.readLine();
					int outputCounter = 0;
		    		while(outputBr != null)
		    		{
			    		Double[] outputTempLine = new Double[2];
				    	String[] outputSplitLine = outputLine.split(",");
				    	Double outputValueToMatch = null;
				    	outputValueToMatch = Double.parseDouble(outputSplitLine[0]);
				    	
				    	
				    	if(valueToMatch.equals(outputValueToMatch))
				    	{
				    		out.println(outputLine);
				    		break;
				    	}
				    	counter++;
				    	outputLine = outputBr.readLine();
		    		}
		    		outputBr.close();
		    	}catch(Exception e)
		    	{
			    	
		    	}
		    		
		    	line = br.readLine();
		    	if(counter % 10 == 0)
		    		System.out.println("Counter at: " + counter);
		    	counter++;
		    }
		    
		    System.out.println(matchesFound);
		    
		    br.close();
		    out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	
	public static HashSet<Double> readSampleSolutionFile()
	{
		HashSet<Double> hs = new HashSet<Double>();
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\sample_submission.csv\\sample_submission.csv");
		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			String line = br.readLine();
			line = br.readLine();
		    
		    while (line != null) {
		    	
		    	Double[] tempLine = new Double[7];
		    	String[] splitLine = line.split(",");
		    	Double tempValueBeforeSanitation = null;
		    	tempValueBeforeSanitation = Double.parseDouble(splitLine[0]);
		    	hs.add(tempValueBeforeSanitation);
		    	line = br.readLine();
		    }
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println(hs.size());
		return hs;
	}
	
	public static void readPropertiesForOutputFile(HashSet<Double> resultsToMatch)
	{
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\properties_2016.csv");

		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			PrintWriter out = new PrintWriter( "C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbySampleOutput.csv" );
			
			String line = br.readLine();
			line = br.readLine();
		    int counter = 0;
		    int matchesFound = 0;
			
		    while (line != null ) {
		    	
		    	Double[] tempLine = new Double[57];
		    	String[] splitLine = line.split(",");
		    	
		    	Double valueToMatch = Double.parseDouble(splitLine[0]);
		    	
		    	if(resultsToMatch.contains(valueToMatch))
		    	{
//		    		System.out.println("Found match: " + splitLine[0]);
		    		matchesFound++;
		    		out.println(line);
		    	}
		    		
		    	line = br.readLine();
		    	if(counter % 100000 == 0)
		    		System.out.println("Counter at: " + counter);
		    	counter++;
		    }
		    
		    System.out.println(matchesFound);
		    
		    br.close();
		    out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	
}
