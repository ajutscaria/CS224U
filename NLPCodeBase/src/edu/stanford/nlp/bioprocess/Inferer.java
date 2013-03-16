package edu.stanford.nlp.bioprocess;

import java.util.List;

public abstract class Inferer {
	public abstract List<BioDatum> BaselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff);
	public abstract List<BioDatum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff);
}
