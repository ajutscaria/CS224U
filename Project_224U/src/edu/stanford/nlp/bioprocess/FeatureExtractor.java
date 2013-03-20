package edu.stanford.nlp.bioprocess;

import java.util.List;
import java.util.Set;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;

public abstract class FeatureExtractor {
	public abstract List<Datum> setFeaturesTrain(List<Example> data);
	public abstract List<Datum> setFeaturesTest(CoreMap sentence, Set<Tree> selectedNodes);
}