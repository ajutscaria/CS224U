package edu.stanford.nlp.bioprocess.joint.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.BioDatum;

public class PrevRelationConstraintGenerator extends ILPConstraintGenerator {

	public PrevRelationConstraintGenerator() {
		super("Yij,prev + Yjk, prev - Yik,prev < = 1", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance all, InferenceVariableLexManager lexicon) {

		BioprocessesInput input = (BioprocessesInput)all;
		List<BioDatum> relationPredicted = input.data;
		
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		int slotlength = input.labels;
		System.out.println("Inside Prev Contradiction");
        System.out.println("relationpredicted size: "+relationPredicted.size());
        HashMap<String, List<Integer>> processToEvent = Inference.processToEvent;
		//relation(i, j) -> event(i) ^ event(j)
        
        for(String process : processToEvent.keySet()){
        	List<Integer> events = processToEvent.get(process);
        	for(int i = 0; i < events.size()-2;i++){
        		for(int j = i+1; j < events.size()-1;j++){
        			for(int k = j+1; k < events.size();k++){
        				//System.out.println(Id1+", "+Id2+", "+Id3);
        				int[] var = new int[3];
    					double[] coef = new double[3];
    					int Id1 = events.get(i);
    					int Id2 = events.get(j);
    					int Id3 = events.get(k);
        				int labelId1 = 2; //PREV
        				int labelId2 = 2; //PREV
        				int labelId3 = 0; //NONE
        				var[0] = lexicon.getVariable(Inference.getVariableName(Id1, Id2, labelId1, "relation"));
    					coef[0] = 1;
    					var[1] = lexicon.getVariable(Inference.getVariableName(Id2, Id3, labelId2, "relation"));
    					coef[1] = 1;
    					var[2] = lexicon.getVariable(Inference.getVariableName(Id1, Id3, labelId3, "relation"));
    					coef[2] = -1;
    					constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.LESS_THAN));
    					
    					labelId1 = 2; //PREV
        				labelId2 = 0; //PREV
        				labelId3 = 2; //NONE
        				var = new int[3];
        				coef = new double[3];
        				var[0] = lexicon.getVariable(Inference.getVariableName(Id1, Id2, labelId1, "relation"));
    					coef[0] = 1;
    					var[1] = lexicon.getVariable(Inference.getVariableName(Id2, Id3, labelId2, "relation"));
    					coef[1] = -1;
    					var[2] = lexicon.getVariable(Inference.getVariableName(Id1, Id3, labelId3, "relation"));
    					coef[2] = 1;
    					constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.LESS_THAN));
    					
    					labelId1 = 0; //PREV
        				labelId2 = 2; //PREV
        				labelId3 = 2; //NONE
        				var = new int[3];
        				coef = new double[3];
        				var[0] = lexicon.getVariable(Inference.getVariableName(Id1, Id2, labelId1, "relation"));
    					coef[0] = -1;
    					var[1] = lexicon.getVariable(Inference.getVariableName(Id2, Id3, labelId2, "relation"));
    					coef[1] = 1;
    					var[2] = lexicon.getVariable(Inference.getVariableName(Id1, Id3, labelId3, "relation"));
    					coef[2] = 1;
    					constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.LESS_THAN));
                	}
            	}
        	}
        }

		return constraints;
	}

	@Override
	public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
			IStructure arg1, InferenceVariableLexManager arg2) {
		// This function looks at the structure that is provided as input, and
		// returns only constraints that are violated by it.

		return new ArrayList<ILPConstraint>();
	}

}

