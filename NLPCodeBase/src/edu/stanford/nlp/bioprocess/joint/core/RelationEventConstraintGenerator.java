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

public class RelationEventConstraintGenerator extends ILPConstraintGenerator {

	public RelationEventConstraintGenerator() {
		super("A => B^C", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance all, InferenceVariableLexManager lexicon) {

		BioprocessesInput input = (BioprocessesInput)all;
		List<BioDatum> relationPredicted = input.data;
		
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		int slotlength = input.labels;
		System.out.println("Inside EventRelation");
        System.out.println("relationpredicted size: "+relationPredicted.size());
        
		//relation(i, j) -> event(i) ^ event(j)
        
        for (int Id = 0; Id < relationPredicted.size(); Id++) {
			for (int labelId = 1; labelId < slotlength; labelId++) { //except "NONE"
				StringBuilder print = new StringBuilder();
				int[] var = new int[2];
				double[] coef = new double[2];
				int event1 = relationPredicted.get(Id).event1_index;
				int event2 = relationPredicted.get(Id).event2_index;
				var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, "relation"));
				coef[0] = -1;
				print.append("-");
				print.append(Inference.getVariableName(event1, event2, labelId, "relation"));
				print.append("+");
				// -A + B >= 0
				var[1] = lexicon.getVariable(Inference.getVariableName(event1, Inference.E_ID, "event"));
				coef[1] = 1;
				constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.GREATER_THAN));
				print.append(Inference.getVariableName(event1, Inference.E_ID, "event"));
				print.append(">=0");
				//System.out.println(print);
				// -A + C >= 0
				var = new int[2];
				coef = new double[2];
				var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, "relation"));
				coef[0] = -1;
				var[1] = lexicon.getVariable(Inference.getVariableName(event2, Inference.E_ID, "event"));
				coef[1] = 1;
				constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.GREATER_THAN));
				print.append(Inference.getVariableName(event2, Inference.E_ID, "event"));
				print.append(">=0");
				//System.out.println(print);
				
				//System.out.println("\n");
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
