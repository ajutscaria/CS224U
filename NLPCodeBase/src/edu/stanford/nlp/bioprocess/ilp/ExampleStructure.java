package edu.stanford.nlp.bioprocess.ilp;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

public class ExampleStructure implements IStructure {

	public final String[] label;
	public BioprocessesInput input;

	public ExampleStructure(BioprocessesInput input, String[] label) {
		this.input = input;
		assert input.slots == label.length;
		
		this.label = label;
	}

	@Override
	public FeatureVector getFeatureVector() {
		// this is where the feature vector for the structure will go.
		throw new RuntimeException("Not implemented!");

	}
}
