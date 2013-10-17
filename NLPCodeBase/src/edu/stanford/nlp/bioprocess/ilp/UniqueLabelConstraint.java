package edu.stanford.nlp.bioprocess.ilp;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;

public class UniqueLabelConstraint extends ILPConstraintGenerator {

	public UniqueLabelConstraint() {
		super("Unique labels", false);

	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance arg0,
			InferenceVariableLexManager lexicon) {

		ExampleInput input = (ExampleInput) arg0;
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		
		for (int slotId = 0; slotId < input.slots; slotId++) {
			
			
			int[] vars = new int[Inference.validLabels.length];
			double[] coefficients = new double[Inference.validLabels.length];

			for (int labelId = 0; labelId < Inference.validLabels.length; labelId++) {
				vars[labelId] = lexicon.getVariable(Inference.getVariableName(slotId,
						labelId));
				coefficients[labelId] = 1.0;
			}

			constraints.add(new ILPConstraint(vars, coefficients, 1.0,
					ILPConstraint.EQUAL));

		}
		return constraints;
	}

	@Override
	public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
			IStructure arg1, InferenceVariableLexManager arg2) {
		return new ArrayList<ILPConstraint>();
	}

}
