package edu.stanford.nlp.bioprocess.ilp;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;

public class ExampleInput implements IInstance {

	public final String name;
	public int slots;

	public ExampleInput(String name, int slots) {
		this.name = name;
		this.slots = slots;
	}

	@Override
	public double size() {
		return 1;
	}

}
