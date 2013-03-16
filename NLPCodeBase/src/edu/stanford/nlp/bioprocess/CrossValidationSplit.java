package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.lang.Math;
/**
 * Split training file to cross validation splits given the total number of folds and the nth fold of test required
 * 
 * @author Rose
 */
public class CrossValidationSplit  {
	ArrayList<Example> allExamplesCV;
	int numFoldsCV;
	List<ArrayList<Example>> foldsCV;
	List<Integer> randomExampleIndex = new ArrayList<Integer>();
	
	public CrossValidationSplit(List<Example> list, int numFolds){
		//for(int i = 0; i < list.size(); i++)
		//	randomExampleIndex.add(i);
		//Collections.shuffle(randomExampleIndex);
		randomExampleIndex = Arrays.asList(new Integer[]{ 42, 20, 46, 40, 32, 2, 39, 1, 30, 5, 21, 34, 31, 19, 36, 27, 29, 24, 28, 23, 18, 45, 9, 13, 7, 15, 10, 48, 26, 3, 16, 6, 12,17, 0, 33, 47, 43, 44, 38, 22, 50, 49, 8, 51, 14, 37, 4, 41, 35, 11, 25});
		//LogInfo.logs(randomExampleIndex);
		numFoldsCV = numFolds;
		allExamplesCV = (ArrayList<Example>)list;
		foldsCV = new ArrayList<ArrayList<Example>>();
		int numSamples = (int) Math.floor(list.size()/numFolds);
		for (int i = 0 ; i < numFolds ; i++){
			ArrayList<Example> devCV = new ArrayList<Example>();
			for (int j = 0 ; j < numSamples ; j++){
				int elemNum = (i * numSamples + j);
				devCV.add(list.get(randomExampleIndex.get(elemNum)));
			}
			foldsCV.add(devCV);
		}
	}
	
	@SuppressWarnings("unchecked")
	public List<Example> GetTrainExamples(int numFold){
		ArrayList<Example> trainCV = new ArrayList<Example>();
		if (numFold<1 || numFold>numFoldsCV){
			return trainCV;
		}
		trainCV = (ArrayList<Example>)allExamplesCV.clone();
		trainCV.removeAll(foldsCV.get(numFold-1));
		return trainCV;
	}
	public List<Example> GetTestExamples(int numFold){
		List<Example> testCV = new ArrayList<Example>();
		if (numFold<1 || numFold>numFoldsCV){
			return testCV;
		}
		return foldsCV.get(numFold-1);
	}

}

