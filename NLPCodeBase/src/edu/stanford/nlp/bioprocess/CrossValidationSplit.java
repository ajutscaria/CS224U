package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.lang.Math;

import fig.basic.LogInfo;
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
		randomExampleIndex = Arrays.asList(new Integer[]{103, 6, 77, 67, 73, 70, 90, 48, 44, 66, 83, 42, 22, 34, 102, 49, 97, 0, 60, 26, 55, 8, 17, 33, 45, 52, 57, 92, 56, 87, 4, 37, 18, 25, 51, 81, 105, 101, 82, 54, 11, 110, 104, 46, 47, 75, 93, 63, 62, 112, 19, 27, 21, 31, 85, 35, 1, 36, 79, 39, 106, 38, 80, 2, 74, 111, 65, 88, 9, 71, 28, 91, 100, 53, 58, 16, 99, 14, 69, 107, 23, 76, 15, 72, 98, 86, 43, 84, 78, 7, 10, 41, 50, 20, 95, 108, 96, 109, 5, 64, 12, 89, 29, 61, 13, 94, 30, 3, 68, 40, 24, 59, 32});
		LogInfo.logs(randomExampleIndex);
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

