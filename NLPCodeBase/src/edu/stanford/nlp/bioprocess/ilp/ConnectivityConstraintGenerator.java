package edu.stanford.nlp.bioprocess.ilp;

import java.util.ArrayList;
import java.util.Arrays;
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
import fig.basic.LogInfo;

public class ConnectivityConstraintGenerator extends ILPConstraintGenerator {

	public ConnectivityConstraintGenerator() {
		super("Zij, Yij, PHIij", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance all,
			InferenceVariableLexManager lexicon) {

		BioprocessesInput input = (BioprocessesInput) all;
		List<BioDatum> relationPredicted = input.data;

		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		int slotlength = input.labels;
		System.out.println("Inside connectivity constraint");
		System.out.println("relationpredicted size: "
				+ relationPredicted.size());
		HashMap<String, List<Integer>> processToEvent = Inference.processToEvent;

		for (int Id = 0; Id < relationPredicted.size(); Id++) {
			int event1 = relationPredicted.get(Id).event1_index;
			int event2 = relationPredicted.get(Id).event2_index;

			// Yij = SUM(Yijr), r != NONE
			int[] vars = new int[slotlength];
			double[] coefficients = new double[slotlength];
			for (int labelId = 1; labelId < slotlength; labelId++) {
				vars[labelId] = lexicon.getVariable(Inference.getVariableName(
						event1, event2, labelId, "relation"));
				coefficients[labelId] = 1.0;
			}
			vars[0] = lexicon.getVariable(Inference.getVariableName(event1,
					event2, "edge", "connectivity")); // Yij
			coefficients[0] = -1;
			constraints.add(new ILPConstraint(vars, coefficients, 0,
					ILPConstraint.EQUAL));

			// equation 2
			int[] var = new int[2];
			double[] coef = new double[2];
			// Zij <= Yij
			var[0] = lexicon.getVariable(Inference.getVariableName(event1,
					event2, "edge", "connectivity")); // Yij
			coef[0] = -1;
			var[1] = lexicon.getVariable(Inference.getVariableName(event1,
					event2, "aux", "connectivity")); // Zij
			coef[1] = 1;
			constraints.add(new ILPConstraint(var, coef, 0,
					ILPConstraint.LESS_THAN));
			// Zji <= Yij
			var = new int[2];
			coef = new double[2];
			var[0] = lexicon.getVariable(Inference.getVariableName(event1,
					event2, "edge", "connectivity")); // Yij
			coef[0] = -1;
			var[1] = lexicon.getVariable(Inference.getVariableName(event2,
					event1, "aux", "connectivity")); // Zji
			coef[1] = 1;
			constraints.add(new ILPConstraint(var, coef, 0,
					ILPConstraint.LESS_THAN));
		}

		for (String process : processToEvent.keySet()) {
			System.out.println("Process:" + process);
			List<Integer> events = processToEvent.get(process);
			/*
			 * for(int e : events){ System.out.println(e); }
			 */
			int size = events.size() - 1;

			int root = events.get(0);
			// equation 3
			System.out.println("equation 3:");
			for (int i = 0; i < events.size(); i++) {
				int[] var = new int[size];
				double[] coef = new double[size];
				int counter = 0;
				int event1 = events.get(i);
				// System.out.println("start");
				for (int j = 0; j < events.size(); j++) {
					int event2 = events.get(j);
					if (i == j)
						continue;
					// System.out.println(Inference.getVariableName(event2,
					// event1, "aux", "connectivity"));
					var[counter] = lexicon.getVariable(Inference
							.getVariableName(event2, event1, "aux",
									"connectivity"));
					coef[counter] = 1;
					counter++;
				}
				if (i == 0) {// root
					constraints.add(new ILPConstraint(var, coef, 0,
							ILPConstraint.EQUAL));
					// System.out.println("=0");
				} else {
					constraints.add(new ILPConstraint(var, coef, 1,
							ILPConstraint.EQUAL));
					// System.out.println("=1");
				}
			}

			// equation 4
			addRootFlowConstraint(lexicon, constraints, events, size, root);

			// equation 5
			addFlowConsistencyConstraints(lexicon, constraints, events, size);

			// equation 6
			// System.out.println("equation 6");
			addFlowEdgeVariableConstraints(lexicon, constraints, events);

		}

		return constraints;
	}

	private void addRootFlowConstraint(InferenceVariableLexManager lexicon,
			List<ILPConstraint> constraints, List<Integer> events, int size,
			int root) {
		int counter = 0;
		int[] var = new int[size];
		double[] coef = new double[size];
		for (int i = 1; i < events.size(); i++) {
			int event2 = events.get(i);

			var[counter] = lexicon.getVariable(Inference.getVariableName(root,
					event2, "flow", "connectivity"));
			coef[counter] = 1;
			counter++;
		}
		constraints
				.add(new ILPConstraint(var, coef, size, ILPConstraint.EQUAL));
	}

	private void addFlowConsistencyConstraints(
			InferenceVariableLexManager lexicon,
			List<ILPConstraint> constraints, List<Integer> events, int size) {
		int counter;
		int[] var;
		double[] coef;
		int doublesize = 2 * size;
		for (int j = 1; j < events.size(); j++) {
			int eventj = events.get(j);
			var = new int[doublesize];
			coef = new double[doublesize];
			counter = 0;
			for (int i = 0; i < events.size(); i++) {
				if (i == j)
					continue;
				int eventi = events.get(i);
				var[counter] = lexicon.getVariable(Inference.getVariableName(
						eventi, eventj, "flow", "connectivity"));
				coef[counter] = 1;
				counter++;
			}
			for (int k = 0; k < events.size(); k++) {
				if (j == k)
					continue;
				int eventk = events.get(k);
				var[counter] = lexicon.getVariable(Inference.getVariableName(
						eventj, eventk, "flow", "connectivity"));
				coef[counter] = -1;
				counter++;
			}
			constraints
					.add(new ILPConstraint(var, coef, 1, ILPConstraint.EQUAL));
		}
	}

	private void addFlowEdgeVariableConstraints(
			InferenceVariableLexManager lexicon,
			List<ILPConstraint> constraints, List<Integer> events) {
		int n = events.size();
		LogInfo.logs("Events=%s", events);
		for (int i = 0; i < events.size(); i++) {
			int eventi = events.get(i);
			
			for (int j = 0; j < events.size(); j++) {
				if (i == j)
					continue;
				int eventj = events.get(j);
				int[] var = new int[2];
				double[] coef = new double[2];
				String phiij = Inference.getVariableName(eventi, eventj,
						"flow", "connectivity");
				var[0] = lexicon.getVariable(phiij); // PHIij
				coef[0] = 1;

				String zij = Inference.getVariableName(eventi, eventj, "aux",
						"connectivity");
				var[1] = lexicon.getVariable(zij); // Zij
				coef[1] = -n;
				/*LogInfo.logs(
						"i=%s, j=%s, eventi=%s, eventj=%s, phiij=%s, zij=%s, var=%s,coeff=%s",
						i, j, eventi, eventj, phiij, zij, Arrays.toString(var),
						Arrays.toString(coef));*/

				// System.out.println("coef:"+coef[1]);
				ILPConstraint constraint = new ILPConstraint(var, coef, 0,
						ILPConstraint.LESS_THAN);

				//LogInfo.logs("Constraint: %s", constraint.toString());
				constraints.add(constraint);
			}
		}
	}

	@Override
	public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
			IStructure arg1, InferenceVariableLexManager arg2) {
		// This function looks at the structure that is provided as input, and
		// returns only constraints that are violated by it.

		return new ArrayList<ILPConstraint>();
	}

}
