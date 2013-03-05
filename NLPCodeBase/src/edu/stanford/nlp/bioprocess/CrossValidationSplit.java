package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
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
	public CrossValidationSplit(ArrayList<Example> examples, int numFolds){
		numFoldsCV = numFolds;
		allExamplesCV = examples;
		foldsCV = new ArrayList<ArrayList<Example>>();
		for (int i = 0 ; i < numFolds ; i++){
			ArrayList<Example> devCV = new ArrayList<Example>();
			for (int j = 0 ; j < examples.size()/numFolds ; j++){
				int elemNum = (i*(int)(Math.floor(examples.size()/numFolds)) + j);
				if (elemNum == examples.size())
					break;
				devCV.add(examples.get(elemNum));
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

