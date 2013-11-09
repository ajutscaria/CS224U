package edu.stanford.nlp.bioprocess.ilp.example;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.stanford.nlp.bioprocess.BioDatum;

public class ExampleInput implements IInstance {

	public final String name;
	public int slots;
	public int labels;
    public List<BioDatum> data;
    public HashMap<Integer, HashSet<Integer>> map;
    public HashMap<Integer, HashSet<Integer>> map2;
    public HashMap<String, List<Integer>> processRelation;
	public ExampleInput(String name, int slots, int labels, List<BioDatum> data) {
		this.name = name;
		this.slots = slots;
		this.labels = labels;
		this.data = data;
	}
	
	public ExampleInput(String name, int slots, int labels, HashMap<Integer, HashSet<Integer>> map) {
		this.name = name;
		this.slots = slots;
		this.labels = labels;
		this.map = map;
	}
	
	public ExampleInput(String name, int slots, int labels, HashMap<Integer, HashSet<Integer>> map, HashMap<Integer, HashSet<Integer>> map2) {
		this.name = name;
		this.slots = slots;
		this.labels = labels;
		this.map = map;
		this.map2 = map2;
	}

	@Override
	public double size() {
		return 1;
	}

}
