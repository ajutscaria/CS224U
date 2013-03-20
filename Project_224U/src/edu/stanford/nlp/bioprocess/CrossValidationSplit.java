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
		randomExampleIndex = Arrays.asList(new Integer[]{ 49, 47, 38, 7, 36, 13, 28, 45, 32, 1, 51, 6, 42, 10, 15, 5, 40, 41, 12, 4, 18, 27, 2, 46, 29, 21, 23, 26, 34, 43, 20, 33, 48, 31, 35, 16, 22, 50, 9, 37, 25, 11, 0, 8, 44, 3, 30, 19, 39, 14, 17, 24});
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

