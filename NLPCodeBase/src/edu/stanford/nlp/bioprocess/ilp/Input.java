package edu.stanford.nlp.bioprocess.ilp;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;

public class Input implements IInstance {

	public final String name;
	public int slots;

	public Input(String name, int slots) {
		this.name = name;
		this.slots = slots;
	}

	@Override
	public double size() {
		return 1;
	}

}
