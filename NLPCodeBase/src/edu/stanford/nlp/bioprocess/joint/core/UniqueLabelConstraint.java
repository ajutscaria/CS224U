package edu.stanford.nlp.bioprocess.joint.core;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.BioDatum;
import edu.stanford.nlp.bioprocess.ilp.example.ExampleInput;


public class UniqueLabelConstraint extends ILPConstraintGenerator {

	public UniqueLabelConstraint() {
		super("Unique labels", false);

	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance arg0,
			InferenceVariableLexManager lexicon) {

		Input input = (Input) arg0;
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		
		addEventUnique(lexicon, input, constraints);
		addEntityUnique(lexicon, input, constraints);	
		addRelaionUnique(lexicon, input, constraints);
		
		return constraints;
	}

	private void addRelaionUnique(InferenceVariableLexManager lexicon,
			Input input, List<ILPConstraint> constraints) {
		for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
			int event1 = input.getEERelationCandidatePair(Id).getSource(); // ?
			int event2 = input.getEERelationCandidatePair(Id).getTarget(); // ?
			int labelLength = Inference.relationLabels.length;
			int[] vars = new int[labelLength];
			double[] coefficients = new double[labelLength];
            
			for (int labelId = 0; labelId < labelLength; labelId++) {
				vars[labelId] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, "relation"));
				coefficients[labelId] = 1.0;
			}
			
			constraints.add(new ILPConstraint(vars, coefficients, 1.0,
					ILPConstraint.EQUAL));
		}
	}

	private void addEntityUnique(InferenceVariableLexManager lexicon,
			Input input, List<ILPConstraint> constraints) {
		for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {	
			for(int entityId = 0; entityId < input.getNumberOfArgumentCandidates(eventId); entityId++){
				int labelLength = Inference.entityLabels.length;
				int[] vars = new int[labelLength];
				double[] coefficients = new double[labelLength];
				for (int labelId = 0; labelId < labelLength; labelId++) {
					vars[labelId] = lexicon.getVariable(Inference.getVariableName(eventId,labelId, "event"));
					coefficients[labelId] = 1.0;
				}
				constraints.add(new ILPConstraint(vars, coefficients, 1.0,
						ILPConstraint.EQUAL));
			}			
		}
	}

	private void addEventUnique(InferenceVariableLexManager lexicon,
			Input input, List<ILPConstraint> constraints) {
		for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
			int labelLength = Inference.eventLabels.length;
			int[] vars = new int[labelLength];
			double[] coefficients = new double[labelLength];
           
			for (int labelId = 0; labelId < labelLength; labelId++) {
				vars[labelId] = lexicon.getVariable(Inference.getVariableName(eventId,labelId, "entity"));
				coefficients[labelId] = 1.0;
			}
			
			constraints.add(new ILPConstraint(vars, coefficients, 1.0,
					ILPConstraint.EQUAL));

		}
	}

	@Override
	public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
			IStructure arg1, InferenceVariableLexManager arg2) {
		return new ArrayList<ILPConstraint>();
	}

}
