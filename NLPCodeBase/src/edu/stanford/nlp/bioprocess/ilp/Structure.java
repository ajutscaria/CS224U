package edu.stanford.nlp.bioprocess.ilp;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

public class Structure implements IStructure {

	public final String[] label;
	public Input input;

	public Structure(Input input, String[] label) {
		this.input = input;
		this.label = label;
	}

	@Override
	public FeatureVector getFeatureVector() {
		// this is where the feature vector for the structure will go.

		return null;
	}
}
