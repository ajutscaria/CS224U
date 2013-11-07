package edu.stanford.nlp.bioprocess.ilp.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;

public class EntityChildrenConstraintGenerator extends ILPConstraintGenerator {

	public EntityChildrenConstraintGenerator() {
		super("A => Exists(B)", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance x, InferenceVariableLexManager lexicon) {
		
		ExampleInput input = (ExampleInput)x;
		HashMap<Integer, HashSet<Integer>> entityChildren = input.map;
        String type = input.name;
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
        System.out.println("entityChildren size: "+entityChildren.size());
		//parent is entity -> child is not entity
		for (Integer parentId : entityChildren.keySet()) {
			int[] var = new int[2];
			double[] coef = new double[2];

			var[0] = lexicon.getVariable(Inference.getVariableName(parentId, Inference.E_ID, type));
			coef[0] = -1;
			//System.out.println(Inference.getVariableName(parentId, Inference.E_ID, type));
			//System.out.println(entityChildren.get(parentId).size());
			
			for(Integer childId : entityChildren.get(parentId)){
				StringBuilder print = new StringBuilder();
				print.append("-");
				print.append(Inference.getVariableName(parentId, Inference.E_ID, type));
				print.append(" + ");
				var[1] = lexicon.getVariable(Inference.getVariableName(childId, Inference.O_ID, type));
				coef[1] = 1;
				print.append(Inference.getVariableName(childId, Inference.O_ID, type));
				print.append(" >=0\n");
				//System.out.println(print);
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
