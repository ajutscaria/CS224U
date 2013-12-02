package edu.stanford.nlp.bioprocess.joint.core;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

/**
 * 
 * @author svivek
 */
public class Structure implements IStructure {
	public final Input input;
	private final String[][] relations;
	private final String[][] arguments;
	private final boolean[] triggers;

	public Structure(Input input, boolean[] triggers, String[][] arguments,
			String[][] relations) {
		this.input = input;
		this.triggers = triggers;
		this.arguments = arguments;
		this.relations = relations;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return super.toString();
	}

	@Override
	public FeatureVector getFeatureVector() {
		// TODO Auto-generated method stub
		return null;
	}

}