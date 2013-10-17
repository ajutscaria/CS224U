package edu.stanford.nlp.bioprocess.ilp;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;

public class ValidAConstraintGenerator extends ILPConstraintGenerator {

	public ValidAConstraintGenerator() {
		super("A => Exists(B)", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance x,
			InferenceVariableLexManager lexicon) {
		ExampleInput input = (ExampleInput) x;

		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();

		// this constraint says that if the label for some slot is A, then there
		// must be some other slot with label B
		// that is, A => exists (B)
		// that is \neg{A} OR B_i

		for (int slotId = 0; slotId < input.slots; slotId++) {
			int[] var = new int[input.slots];
			double[] coef = new double[input.slots];

			// the first var in the constraint says that slotId -> A. The other vars
			// say that other slots map to B

			var[0] = lexicon.getVariable(Inference.getVariableName(slotId,
					Inference.A_ID));
			coef[0] = -1;

			int i = 1;
			for (int otherSlotId = 0; otherSlotId < input.slots; slotId++) {
				if (otherSlotId == slotId)
					continue;
				var[i] = lexicon.getVariable(Inference.getVariableName(otherSlotId,
						Inference.B_ID));
				coef[i] = -1;
				i++;

			}

			constraints.add(new ILPConstraint(var, coef, 0,
					ILPConstraint.GREATER_THAN));
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
