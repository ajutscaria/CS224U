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
	
	public CrossValidationSplit(ArrayList<Example> examples, int numFolds){
		for(int i = 0; i < examples.size(); i++)
			randomExampleIndex.add(i);
		//Collections.shuffle(randomExampleIndex);
		randomExampleIndex = Arrays.asList(new Integer[]{51, 1, 30, 5, 21, 28, 23, 18, 45, 42, 20, 46, 40, 32, 2, 39, 17, 0, 33, 47, 43, 44, 38, 22, 50, 49, 8, 9, 13, 7, 15, 10, 48, 26, 3, 16, 6, 12, 14, 37, 4, 41, 35, 34, 31, 19, 36, 27, 29, 24, 11, 25});
		LogInfo.logs(randomExampleIndex);
		numFoldsCV = numFolds;
		allExamplesCV = examples;
		foldsCV = new ArrayList<ArrayList<Example>>();
		for (int i = 0 ; i < numFolds ; i++){
			ArrayList<Example> devCV = new ArrayList<Example>();
			for (int j = 0 ; j < examples.size()/numFolds ; j++){
				int elemNum = (i*(int)(Math.floor(examples.size()/numFolds)) + j);
				devCV.add(examples.get(randomExampleIndex.get(elemNum)));
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

