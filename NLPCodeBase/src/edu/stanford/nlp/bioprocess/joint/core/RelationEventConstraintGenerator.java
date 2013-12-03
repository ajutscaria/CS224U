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

		Input input = (Input)all;
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
			int event1 = input.getEERelationCandidatePair(Id).getSource(); // ?
			int event2 = input.getEERelationCandidatePair(Id).getTarget(); // ?
			int labelLength = Inference.relationLabels.length;
			
			for (int labelId = 1; labelId < labelLength; labelId++) { //excluding NONE
				int[] var = new int[2];
				double[] coef = new double[2];
				var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, "relation"));
				coef[0] = -1;
				var[1] = lexicon.getVariable(Inference.getVariableName(event1, Inference.E_ID, "event"));
				coef[1] = 1;
				constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.GREATER_THAN));
				
				var = new int[2];
				coef = new double[2];
				var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, "relation"));
				coef[0] = -1;
				var[1] = lexicon.getVariable(Inference.getVariableName(event2, Inference.E_ID, "event"));
				coef[1] = 1;
				constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.GREATER_THAN));
				
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
