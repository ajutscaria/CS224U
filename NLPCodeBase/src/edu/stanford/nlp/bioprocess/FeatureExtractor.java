package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.util.CoreMap;

public abstract class FeatureExtractor {
	public abstract List<Datum> setFeaturesTrain(List<Example> data);
	public abstract List<Datum> setFeaturesTest(CoreMap sentence);
}