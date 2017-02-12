import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;


public class ImageHelper {
	int max = 255;
	String directoryPath;
	
	public static void main(String[] args)
	{
		ImageHelper helper = new ImageHelper("C:\\Users\\Darby\\OneDrive\\Documents\\");
		Double[] array = helper.extractBytes("eight3.png");
		for(Double d : array)
		{
			System.out.print(d + ", ");
		}
		
	}
	public ImageHelper(String directoryPath){
		this.directoryPath = directoryPath;
	}
	
	public Double[] extractBytes (String ImageName) {
		 // open image
		
		File imgPath = new File(directoryPath + ImageName);
		BufferedImage bufferedImage = null;
		try {
			bufferedImage = ImageIO.read(imgPath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int[][] result = null ;
		
		for (int i = 0; i < 10; i++) {
			result = convertTo2DWithoutUsingGetRGB(bufferedImage);
		}
		
		Double[] returnArray = new Double[27*27];
		int m = result.length;
		int n = result[0].length;
		for(int i = 0; i < m; i++)
		{
			for(int j = 0; j < n; j++)
			{
				returnArray[n*i + j] = result[i][j]/255.;
			}
		}
		 
		return returnArray;
	}
	
	  private int[][] convertTo2DWithoutUsingGetRGB(BufferedImage image) {

	      final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	      final int width = image.getWidth();
	      final int height = image.getHeight();
	      final boolean hasAlphaChannel = image.getAlphaRaster() != null;

	      int[][] result = new int[height][width];
	      if (hasAlphaChannel) {
	         final int pixelLength = 4;
	         for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
	            int argb = 0;
//	            argb += (((int) pixels[pixel] & 0xff) << 24); // alpha
	            argb += ((int) pixels[pixel + 1] & 0xff); // blue
//	            argb += (((int) pixels[pixel + 2] & 0xff) << 8); // green
//	            argb += (((int) pixels[pixel + 3] & 0xff) << 16); // red
	            result[row][col] = argb;
	            col++;
	            if (col == width) {
	               col = 0;
	               row++;
	            }
	         }
	      } else {
	         final int pixelLength = 3;
	         for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
	            int argb = 0;
	            argb += -16777216; // 255 alpha
	            argb += ((int) pixels[pixel] & 0xff); // blue
	            argb += (((int) pixels[pixel + 1] & 0xff) << 8); // green
	            argb += (((int) pixels[pixel + 2] & 0xff) << 16); // red
	            result[row][col] = argb;
	            col++;
	            if (col == width) {
	               col = 0;
	               row++;
	            }
	         }
	      }

	      return result;
	   }
}
