package edu.stanford.nlp.bioprocess.ilp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import LBJ2.infer.ILPSolver;
import edu.illinois.cs.cogcomp.infer.ilp.AbstractILPInference;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;

public class Inference extends AbstractILPInference<Structure> {

	public static final String[] validLabels = { "A", "B", "C", "D", "E" };

	// the id for A & B in the above list. This can be done better
	public static final int A_ID = 0;
	public final static int B_ID = 1;

	private List<ILPConstraintGenerator> constraints;
	private Input input;

	public Inference(Input input, ILPSolverFactory solverFactory, boolean debug) {
		super(solverFactory, debug);
		this.input = input;

		constraints = new ArrayList<ILPConstraintGenerator>();
		constraints.add(new UniqueLabelConstraint());
		constraints.add(new ValidAConstraintGenerator());
	}

	@Override
	protected void addConstraints(ILPSolver solver,
			InferenceVariableLexManager lexicon) {
		for (ILPConstraintGenerator c : constraints) {
			for (ILPConstraint constraint : c.getILPConstraints(input, lexicon))
				this.addConstraint(solver, constraint);
		}
	}

	@Override
	protected void addVariables(ILPSolver solver,
			InferenceVariableLexManager lexicon) {
		// each label can take one of the five values

		for (int slotId = 0; slotId < input.slots; slotId++) {
			for (int labelId = 0; labelId < validLabels.length; labelId++) {
				double score = getLabelScore(slotId, labelId);

				int var = solver.addBooleanVariable(score);
				String varName = getVariableName(slotId, labelId);

				lexicon.addVariable(varName, var);
			}
		}
	}

	public static String getVariableName(int slotId, int labelId) {
		return "slot" + slotId + "-" + labelId;
	}

	private double getLabelScore(int slotId, int labelId) {
		// for now, some random scores
		return (new Random()).nextDouble();
	}

	@Override
	protected Structure getOutput(ILPSolver solver,
			InferenceVariableLexManager lexicon) throws Exception {

		String[] labels = new String[input.slots];
		for (int slotId = 0; slotId < input.slots; slotId++) {
			for (int labelId = 0; labelId < validLabels.length; labelId++) {
				String varName = getVariableName(slotId, labelId);

				int var = lexicon.getVariable(varName);

				if (solver.getBooleanValue(var)) {
					labels[slotId] = validLabels[labelId];
				}
			}
		}

		return new Structure(input, labels);
	}
}
