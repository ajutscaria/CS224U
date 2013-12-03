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

		BioprocessesInput input = (BioprocessesInput) arg0;
		List<BioDatum> data = input.data; 
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		int labelLength = input.labels;
		if(!input.name.equals("relation")){
			for (int eventId = 0; eventId < data.size(); eventId++) {
				int[] vars = new int[labelLength];
				double[] coefficients = new double[labelLength];
	            StringBuilder print = new StringBuilder();
				for (int labelId = 0; labelId < labelLength; labelId++) {
					vars[labelId] = lexicon.getVariable(Inference.getVariableName(eventId,labelId, input.name));
					coefficients[labelId] = 1.0;
					print.append(Inference.getVariableName(eventId,labelId, input.name));
					print.append("+");
				}
	
				print.append("= 1\n");
				//System.out.println(print);
				constraints.add(new ILPConstraint(vars, coefficients, 1.0,
						ILPConstraint.EQUAL));
	
			}
		}else{
			for (int eventId = 0; eventId < data.size(); eventId++) { 
				int event1 = data.get(eventId).event1_index;
				int event2 = data.get(eventId).event2_index;
				int[] vars = new int[labelLength];
				double[] coefficients = new double[labelLength];
	            StringBuilder print = new StringBuilder();
				for (int labelId = 0; labelId < labelLength; labelId++) {
					vars[labelId] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, input.name));
					coefficients[labelId] = 1.0;
					print.append(Inference.getVariableName(event1, event2, labelId, input.name));
					print.append("+");
				}
	
				print.append("= 1\n");
				//System.out.println(print);
				constraints.add(new ILPConstraint(vars, coefficients, 1.0,
						ILPConstraint.EQUAL));
	
			}
			
		}
		
		return constraints;
	}

	@Override
	public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
			IStructure arg1, InferenceVariableLexManager arg2) {
		return new ArrayList<ILPConstraint>();
	}

}
