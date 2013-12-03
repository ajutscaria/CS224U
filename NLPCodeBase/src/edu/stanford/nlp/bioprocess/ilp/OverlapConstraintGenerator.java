package edu.stanford.nlp.bioprocess.ilp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;


public class OverlapConstraintGenerator extends ILPConstraintGenerator {

	public OverlapConstraintGenerator() {
		super("A => Exists(B)", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance x, InferenceVariableLexManager lexicon) {
		
		BioprocessesInput input = (BioprocessesInput)x;
		HashMap<Integer, HashSet<Integer>> entityOverlapEvent = input.map;
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
        
		//entity -> overlapped event candidates should not be triggers
		for (Integer parentId : entityOverlapEvent.keySet()) {
			
			//System.out.println(Inference.getVariableName(parentId, Inference.E_ID, type));
			//System.out.println(entityChildren.get(parentId).size());
			
			for(Integer childId : entityOverlapEvent.get(parentId)){
				int[] var = new int[2];
				double[] coef = new double[2];

				var[0] = lexicon.getVariable(Inference.getVariableName(parentId, Inference.E_ID, "entity"));
				coef[0] = -1;
				
				var[1] = lexicon.getVariable(Inference.getVariableName(childId, Inference.O_ID, "event"));
				coef[1] = 1;

				constraints.add(new ILPConstraint(var, coef, 0,
						ILPConstraint.GREATER_THAN));
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

