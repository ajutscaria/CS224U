package edu.stanford.nlp.bioprocess;

import java.util.List;

public abstract class Inferer {
	public abstract List<Datum> BaselineInfer(List<Example> examples, Params parameters);
	public abstract List<Datum> Infer(List<Example> testData, Params parameters);
}
