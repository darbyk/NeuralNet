import java.util.Arrays;

public class NeuralMatrix {

    // return a random m-by-n matrix with values between 0 and 1
    public static double[][] random(int m, int n) {
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
            	int x = (int)(Math.random()*1000);
                a[i][j] = x/1000.0;
            }
        return a;
    }

    // return n-by-n identity matrix I
    public static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            a[i][i] = 1;
        return a;
    }

    // return x^T y
    public static double dot(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // return B = A^T
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[j][i] = a[i][j];
        return b;
    }
    

	public static double[][] transpose(double[] a) {
        int m = a.length;
//        int n = a[0].length;
        double[][] b = new double[m][1];
        for (int i = 0; i < m; i++)
                b[i][0] = a[i];
        return b;
	}
    

    // return c = a + b
    public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] + b[i][j];
        return c;
    }

    // return c = a - b
    public static Double[][] subtract(Double[][] y, Double[][] yHat) {
        int m = y.length;
        int n = y[0].length;
        Double[][] c = new Double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
//            	System.out.println(a[i][j] - b[i][j]);
                c[i][j] = y[i][j] - yHat[i][j];
            }
        return c;
    }

    // return c = a * b
    public static Double[][] multiply(Double[][] a, Double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        Double[][] c = new Double[m1][n2];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
                for (int k = 0; k < n1; k++)
                {
                	if(c[i][j] == null)
                		c[i][j] = a[i][k] * b[k][j];
                	else
                		c[i][j] += a[i][k] * b[k][j];
                }
        return c;
    }
    
    // return c = a * b
    public static Double[][] multiplyVector(Double[][] costResultMatrix, Double[][] costResultMatrix2) {
        int m1 = costResultMatrix.length;
        int n1 = costResultMatrix[0].length;
        int m2 = costResultMatrix2.length;
        int n2 = costResultMatrix2[0].length;
        if (n2 != 1) throw new RuntimeException("Illegal matrix dimensions. Matrix B must be of size m x 1");
        if (m1 != m2) throw new RuntimeException("Illegal matrix dimensions. m1: " + m1 + " m2: " + m2);
        Double[][] c = new Double[m1][n2];
        for (int i = 0; i < m1; i++)
                for (int k = 0; k < n1; k++)
                {
//                	System.out.println("a @ (" + i + "," + k + ")");
//                	System.out.println("b @ (" + i + "," + 0 + ")");
//                	System.out.println(a[i][k] + "*" + b[i][0]);
                	if(c[i][0] == null)
                		c[i][0] = costResultMatrix[i][k] * costResultMatrix2[i][0];
                	else
                		c[i][0] += costResultMatrix[i][k] * costResultMatrix2[i][0];
                }
        return c;
    }
    
    // return c = a * b
    public static double[][] multiplyScalar(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
//        if (n2 != 1) throw new RuntimeException("Illegal matrix dimensions. Matrix B must be of size m x 1");
        if (m1 != m2) throw new RuntimeException("Illegal matrix dimensions. m1: " + m1 + " m2: " + m2);
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++)
                for (int k = 0; k < n1; k++)
                {
//                	System.out.println("a @ (" + i + "," + k + ")");
//                	System.out.println("b @ (" + i + "," + 0 + ")");
//                	System.out.println(a[i][k] + "*" + b[i][0]);
                    c[i][k] += a[i][k] * b[i][k];
                }
        return c;
    }

    
    // return c = A - epsilon
    public static double[][] perturbMatrix(double[][] a, double epsilon) {
        int m1 = a.length;
        int n1 = a[0].length;
        double[][] c = new double[m1][n1];
        for (int i = 0; i < m1; i++)
                for (int k = 0; k < n1; k++)
                {
                    c[i][k] += a[i][k] * epsilon;
                }
        return c;
    }
    
    // matrix-vector multiplication (y = A * x)
    public static double[] multiply(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        
//        System.out.println(Arrays.deepToString(a));
//        System.out.println(Arrays.deepToString(a));
		System.out.println();
        
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += a[i][j] * x[j];
        return y;
    }
    
    // Sum Vector
    public static double sumVector(Double[][] costResultMatrix) {
    	//System.out.println(Arrays.deepToString(a));
        int m1 = costResultMatrix.length;
        int n1 = costResultMatrix[0].length;
        //if (n1 != m1) throw new RuntimeException("Illegal matrix dimensions.");
        double c = 0;
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n1; j++)
                    c += costResultMatrix[i][j];
        return c;
    }
    
    // apply Sigmoid function
    public static Double[][] applySigmoid(Double[][] z) {
    	//System.out.println(Arrays.deepToString(a));
        int m1 = z.length;
        int n1 = z[0].length;
        //if (n1 != m1) throw new RuntimeException("Illegal matrix dimensions.");
        Double[][] c = new Double[m1][n1];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n1; j++)
                    c[i][j] = 1/(1 + Math.pow(Math.E, (-1)*(z[i][j]) ));
        return c;
    }
    
    public static double[][] applySigmoidPrime(double[][] a) {
    	//System.out.println(Arrays.deepToString(a));
        int m1 = a.length;
        int n1 = a[0].length;
        //if (n1 != m1) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n1];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n1; j++)
            {
            	double eRaised = Math.pow(Math.E, (-1)*(a[i][j]));
                c[i][j] = eRaised/Math.pow((1 + eRaised), 2);
            }
        return c;
    }


    // vector-matrix multiplication (y = x^T A)
    public static double[] multiply(double[] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += a[i][j] * x[i];
        return y;
    }

    // test client
    public static void main(String[] args) {
        System.out.println("D");
        System.out.println("--------------------");
        double[][] d = { { 1, 2, 3 }, { 4, 5, 6 }, { 9, 1, 3} };
        System.out.print(Arrays.deepToString(d));
        System.out.println();

//        System.out.println("I");
//        System.out.println("--------------------");
//        double[][] c = Matrix.identity(5);
//        System.out.print(Arrays.deepToString(c));
//        System.out.println();
//
        System.out.println("A");
        System.out.println("--------------------");
        double[][] a = Matrix.random(3, 3);
        System.out.print(Arrays.deepToString(a));
        System.out.println();
        System.out.println();

//        System.out.println("A^T");
//        System.out.println("--------------------");
//        double[][] b = Matrix.transpose(a);
//        System.out.print(Arrays.deepToString(b));
//        System.out.println();
//
//        System.out.println("A + A^T");
//        System.out.println("--------------------");
//        double[][] e = Matrix.add(a, b);
//        System.out.print(Arrays.deepToString(e));
//        System.out.println();

//        System.out.println("A * A^T");
//        System.out.println("--------------------");
//        double[][] f = Matrix.multiply(a, b);
//        System.out.print(Arrays.deepToString(f));
//        System.out.println();

        System.out.println("A, sigmoidFunction");
        System.out.println("--------------------");
        double[][] g = Matrix.applySigmoid(a);
        System.out.print(Arrays.deepToString(g));
        System.out.println();
    }

}
