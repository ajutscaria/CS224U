package edu.stanford.nlp.bioprocess.ilp;

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory.SolverType;

public class ILPSolverExample {

	public void run() throws Exception {
		Input input = new Input("some input", 10);

		ILPSolverFactory solverFactory = new ILPSolverFactory(
				SolverType.CuttingPlaneGurobi);

		Inference inference = new Inference(input, solverFactory, false);

		Structure output = inference.runInference();
	}
}
