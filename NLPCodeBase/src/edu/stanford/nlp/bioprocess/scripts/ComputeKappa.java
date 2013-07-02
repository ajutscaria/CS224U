package edu.stanford.nlp.bioprocess.scripts;

public class ComputeKappa {
	public static void main(String args[]){
        double kappa[][] = new double[3][3];
        kappa[0][0]=25;
        kappa[0][1]=35;
        kappa[1][0]=5;
        kappa[1][1]=35;
        
        findKappa(kappa);
    }
    public static void findKappa(double [][]kappa){
        //we have the counts stored in an nxn matrix
        //convert "observed matrix" from counts to frequencies
        double sum = sumOfAllElements(kappa);
        double diagonal = sumDiagonal(kappa);
        double chance = 0.0;
        for(int i = 0; i < kappa.length; ++i) {
        	chance+= (array_sum(getRow(kappa, i))*array_sum(getColumn(kappa, i))) / sum; 
        }
        double kappaValue = (diagonal-chance)/(sum-chance);
        System.out.println("here is your kappa value:"+kappaValue);
    }
    
    private static double sumDiagonal(double[][] kappa) {
    	double sum=0.0;
    	for(int i = 0; i < kappa.length; ++i) 
    		sum+=kappa[i][i];
    	return sum;
	}
	static double array_sum(double arr[]){
        double sum=0;
        for (int i=0; i < arr.length; i++)
            sum+=arr[i];
        return sum;
    }
    static double sumOfAllElements(double[][] matrix){
        double sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                sum+=matrix[i][j];
            }
        }
        return sum;
    }
    static double[] getRow(double matrix[][],int row){
        return matrix[row];
    }
    static double[] getColumn(double matrix[][],int column){
        double col[]=new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
                col[i]=matrix[i][column];
        }
        return col;
    }
}
