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
		for(int i = 0; i < list.size(); i++)
			randomExampleIndex.add(i);
		Collections.shuffle(randomExampleIndex);
		randomExampleIndex = Arrays.asList(new Integer[]{10, 45, 15, 27, 4, 9, 24, 42, 2, 31, 32, 38, 23, 14, 47, 11, 13, 29, 36, 37, 25, 41, 18, 3, 28, 7, 46, 49, 16, 19, 17, 0, 51, 35, 8, 33, 1, 5, 30, 21, 39, 48, 44, 20, 43, 22, 12, 50, 6, 26, 34, 40});
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

