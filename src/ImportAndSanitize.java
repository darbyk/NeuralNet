import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import com.darby.neuralnet.NeuralNet;

public class ImportAndSanitize {

	public static int minCounter = 40000;
	public static int maxCounter = 90000;
	
	public static int propertyFeatures = 59;
	
	public static void main(String args[])
	{
		Double importedValues[][] = loadInputFile("darbyMapping.csv");
		Double importedTrainValues[][] = loadTrainFile("darbyMappingOutput.csv");
	}
	
	public static void createTestOutput(NeuralNet NN)
	{
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbySampleOutput.csv");
		
		//Read from file
		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			PrintWriter out = new PrintWriter( "C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\darbyRealOutput.csv" );
			out.println("ParcelId,201610,201611,201612,201710,201711,201712");
			
			String line = br.readLine();
			line = br.readLine();
			int counter = 1;
		    while (line != null) {
		    	if(counter%100 == 0)
		    		System.out.println(counter);
		    	counter++;
		    	Double[] tempLine = new Double[propertyFeatures];
		    	String[] splitLine = line.split(",");
		    	
		    	String outputString = splitLine[0];
		    	outputString += ",";
		    	
		    	for(int i = 0; i < tempLine.length; i++)
			    {
		    		Double tempValueBeforeSanitation = 0.0;
		    		try
		    		{
		    			tempValueBeforeSanitation = Double.parseDouble(splitLine[i]);
		    		} catch(Exception e) {
		    		}
		    		
		    		if(tempValueBeforeSanitation == null)
		    			tempValueBeforeSanitation = 0.0;
		    		
		    		sanitizeInput(i, tempValueBeforeSanitation, tempLine);
			    }
		    	
		    	Double[][] finalNNTest = new Double[3][propertyFeatures];
		    	
//		    	finalNNTest = new Double[1][propertyFeatures];
		    	
		    	finalNNTest[0] = tempLine.clone();
		    	finalNNTest[1] = tempLine.clone();
		    	finalNNTest[2] = tempLine.clone();
//		    	finalNNTest[3] = tempLine;
//		    	finalNNTest[4] = tempLine;
//		    	finalNNTest[5] = tempLine;
		    	
		    	finalNNTest[0][propertyFeatures - 1] = 10.;
		    	finalNNTest[1][propertyFeatures - 1] = 11.;
		    	finalNNTest[2][propertyFeatures - 1] = 12.;
//		    	finalNNTest[3][propertyFeatures - 1] = 10.;
//		    	finalNNTest[4][propertyFeatures - 1] = 11.;
//		    	finalNNTest[5][propertyFeatures - 1] = 12.;
		    	
		    	
		    	
		    	finalNNTest = NN.normalizeMatrix(finalNNTest);
		    	
		    	NN.setInputData(finalNNTest);
		    	
		    	NN.calculateForwardProp();
		    	
		    	Double[][] resultToWrite = NN.unMinMaxNormalize(NN.getYHat(), NN.outputMin, NN.outputMax);
//		    	resultToWrite[0][0] = Math.round(resultToWrite[0][0] * 10000) / 10000.0;
		    	outputString += Double.toString(Math.round(resultToWrite[0][0] * 10000) / 10000.0) + "," + 
		    					Double.toString(Math.round(resultToWrite[1][0] * 10000) / 10000.0) + "," + 
		    					Double.toString(Math.round(resultToWrite[2][0] * 10000) / 10000.0) + "," + 
		    					Double.toString(Math.round(resultToWrite[0][0] * 10000) / 10000.0) + "," + 
		    					Double.toString(Math.round(resultToWrite[1][0] * 10000) / 10000.0) + "," + 
		    					Double.toString(Math.round(resultToWrite[2][0] * 10000) / 10000.0);
		    	
		    	out.println(outputString);
		        line = br.readLine();
		    }
		    out.flush();
		    out.close();
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch(Exception e) {
			System.out.println("Generic Exception");
			e.printStackTrace();
		}
		
		
	}
	
	public static Double[][] loadInputFile(String inputFileName)
	{
		ArrayList<Double[]> initialImportedValues = new ArrayList<Double[]>();
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\" + inputFileName);
		
		//Read from file
		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			String line = br.readLine();
			line = br.readLine();
		    int counter = 0;
		    while (line != null && counter < maxCounter) {
		    	counter++;
		    	if(counter >= minCounter)
		    	{
			    	Double[] tempLine = new Double[propertyFeatures];
			    	String[] splitLine = line.split(",");
			    	for(int i = 0; i < splitLine.length; i++)
				    {
			    		Double tempValueBeforeSanitation = 0.0;
			    		try
			    		{
			    			tempValueBeforeSanitation = Double.parseDouble(splitLine[i]);
			    		} catch(Exception e) {
			    		}
			    		
			    		if(tempValueBeforeSanitation == null)
			    			tempValueBeforeSanitation = 0.0;
			    		
			    		sanitizeInput(i, tempValueBeforeSanitation, tempLine);
	
				    }
			    	
			    	initialImportedValues.add(tempLine);
		    	}
		    	
		        line = br.readLine();
		    }
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Double importedValues[][] = new Double[initialImportedValues.size()][57];
		for(int i = 0; i < initialImportedValues.size(); i++)
		{
			importedValues[i] = initialImportedValues.get(i);
		}
		
		return importedValues;
		
	}
	
	private static void sanitizeInput(int locationOfSwitch, double tempValueBeforeSanitation, Double[] tempLine)
	{
		int i = locationOfSwitch;
		
		switch(i){
			case 0:  
				// parcelid
				tempLine[i] = 1.0;
				break;
			case 1:  
				// airconditioningtypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 2:
				// architecturalstyletypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 3:
				// basementsqft
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 4:  
				// bathroomcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 5:  
				// bedroomcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 6:  
				// buildingclasstypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 7:  
				// buildingqualitytypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 8:  
				// calculatedbathnbr
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 9:  
				// decktypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 10:  
				// finishedfloor1squarefeet
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 11:  
				// calculatedfinishedsquarefeet
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 12:  
				// finishedsquarefeet12
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 13:  
				// finishedsquarefeet13
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 14:  
				// finishedsquarefeet15
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 15:  
				// finishedsquarefeet50
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 16:  
				//finishedsquarefeet6
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 17:  
				// fips
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 18:  
				// fireplacecnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 19:  
				// fullbathcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 20:  
				// garagecarcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 21:  
				// garagetotalsqft
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 22: 
				// hashottuborspa
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 23:  
				//heatingorsystemtypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 24:  
				// latitude
				tempLine[i] = 1.0;
				break;
			case 25:  
				// longitude
				tempLine[i] = 1.0;
				break;
			case 26:  
				// lotsizesquarefeet
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 27:  
				// poolcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 28:  
				// poolsizesum
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 29:  
				// pooltypeid10
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 30:  
				// pooltypeid2
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 31: 
				// pooltypeid7
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 32:  
				// propertycountylandusecode
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 33:  
				// propertylandusetypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 34:  
				// propertyzoningdesc
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 35:  
				// rawcensustractandblock
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 36:  
				// regionidcity
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 37:  
				// regionidcounty
//				tempLine[i] = 1.0;
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 38:  
				// regionidneighborhood
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 39:  
				// regionidzip
				// want to convert to an aggregate, but will 1 out for now
				tempLine[i] = 1.0;
//				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 40:  
				// roomcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 41:  
				// storytypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 42:  
				// threequarterbathnbr
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 43:  
				// typeconstructiontypeid
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 44:  
				// unitcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 45:  
				// yardbuildingsqft17
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 46:  
				// yardbuildingsqft26
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 47:  
				// yearbuilt
				// Convert to age
//				tempLine[i] = 2017 - tempValueBeforeSanitation;
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 48:  
				// numberofstories
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 49:  
				// fireplaceflag
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 50:  
				// structuretaxvaluedollarcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 51:  
				// taxvaluedollarcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 52:  
				// assessmentyear
				// convert to age
//				tempLine[i] = 2017 - tempValueBeforeSanitation;
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 53:  
				// landtaxvaluedollarcnt
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 54:  
				// taxamount
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 55:  
				// taxdelinquencyflag
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 56:  
				// taxdelinquencyyear
//				tempLine[i] = 2017 - tempValueBeforeSanitation;
				tempLine[i] = tempValueBeforeSanitation;
				break;
			case 57:  
				// censustractandblock
				tempLine[i] = 1.0;
				break;
			case 58:
				// month
				tempLine[i] = tempValueBeforeSanitation;
				break;
		}
	}

	public static Double[][] loadTrainFile(String trainFileName)
	{
		ArrayList<Double[]> initialImportedValues = new ArrayList<Double[]>();
		File importedValuesPath = new File("C:\\Users\\darby.kidwell\\Downloads\\properties_2016.csv\\" + trainFileName);
		
		//Read from file
		try(BufferedReader br = new BufferedReader(new FileReader(importedValuesPath))) {
		    
			String line = br.readLine();
			line = br.readLine();
		    int counter = 0;
		    while (line != null && counter < maxCounter) {
		    	counter++;
		    	if(counter >= minCounter)
		    	{
			    	Double[] tempLine = new Double[1];
			    	String[] splitLine = line.split(",");
			    	
			    	Double tempValueBeforeSanitation = 0.0;
		    		try
		    		{
		    			tempValueBeforeSanitation = Double.parseDouble(splitLine[1]);
		    		} catch(Exception e) {
		    		}
		    		tempLine[0] = tempValueBeforeSanitation; 
			    	initialImportedValues.add(tempLine);
		    	}
		        line = br.readLine();
		    }
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Double importedValues[][] = new Double[initialImportedValues.size()][57];
		for(int i = 0; i < initialImportedValues.size(); i++)
		{
			importedValues[i] = initialImportedValues.get(i);
		}
		
		// Print out our imported values
//		System.out.println("parcelid, airconditioningtypeid, architecturalstyletypeid, basementsqft, bathroomcnt, bedroomcnt, buildingclasstypeid, buildingqualitytypeid, calculatedbathnbr, decktypeid, finishedfloor1squarefeet, calculatedfinishedsquarefeet, finishedsquarefeet12, finishedsquarefeet13, finishedsquarefeet15, finishedsquarefeet50, finishedsquarefeet6, fips, fireplacecnt, fullbathcnt, garagecarcnt, garagetotalsqft, hashottuborspa, heatingorsystemtypeid, latitude, longitude, lotsizesquarefeet, poolcnt, poolsizesum, pooltypeid10, pooltypeid2, pooltypeid7, propertycountylandusecode, propertylandusetypeid, propertyzoningdesc, rawcensustractandblock, regionidcity, regionidcounty, regionidneighborhood, regionidzip, roomcnt, storytypeid, threequarterbathnbr, typeconstructiontypeid, unitcnt, yardbuildingsqft17, yardbuildingsqft26, yearbuilt, numberofstories, fireplaceflag, structuretaxvaluedollarcnt, taxvaluedollarcnt, assessmentyear, landtaxvaluedollarcnt, taxamount, taxdelinquencyflag, taxdelinquencyyear, censustractandblock");
//		for(int i = 0; i < importedValues.length; i++)
//		{
//			System.out.print(i + ":\t");
//			for(int j = 0; j < importedValues[0].length; j++)
//			{
//				System.out.print(importedValues[i][j]);
//				if( j != importedValues[0].length - 1)
//					System.out.print(", ");
//			}
//			System.out.println();
//		}
		
		return importedValues;
		
	}

}
