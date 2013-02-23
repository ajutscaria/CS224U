package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

public class TriggerPredictionInference {
	public void baselineInfer(List<Example> examples) {
		System.out.println("printing CV");
		List<Example> tempList = new ArrayList<Example>();
		for (int i = 0 ; i < 13 ; i++){
			tempList.add(examples.get(i));
			System.out.println(examples.get(i).id);
		}
		System.out.println("printing CV train for fold 4");
		CrossValidationSplit CV = new CrossValidationSplit(tempList,4);
		List<Example> train = CV.GetTrainExamples(4);
		for (Example t:train){
			System.out.println(t.id);
		}
		System.out.println("printing CV test for fold 1");
		List<Example> test = CV.GetTestExamples(1);
		for (Example t:test){
			System.out.println(t.id);
		}
		
		//for(Example example:examples) {
			
		//}
	}
}
