package edu.stanford.nlp.bioprocess.joint.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
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
	public List<ILPConstraint> getILPConstraints(IInstance all, InferenceVariableLexManager lexicon) {

		BioprocessesInput input = (BioprocessesInput)all;
		HashMap<Integer, HashSet<Integer>> eventToEntity = input.map;
		HashMap<Integer, HashSet<Integer>> entityToEvent = input.map2;
		
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		System.out.println("Inside Valid");
        System.out.println("eventToEntity size: "+eventToEntity.size());
        //System.out.println("eventToEntity size: "+eventToEntity.keySet().size());
		//non-event -> non-entity
		for (Integer eventId : eventToEntity.keySet()) {
			
			//System.out.println(Inference.getVariableName(eventId, Inference.O_ID, "event"));
			//System.out.println(eventToEntity.get(eventId).size());
			//each event has at least one entity
			int numEntities = eventToEntity.get(eventId).size() ;
			int[] vars = new int[numEntities + 1];
			double[] coefs = new double[numEntities + 1];			
			vars[0] = lexicon.getVariable(Inference.getVariableName(eventId, Inference.E_ID, "event"));
			coefs[0] = -1;
			int counter = 1;
			for(Integer entityId : eventToEntity.get(eventId)){
				int[] var = new int[2];
				double[] coef = new double[2];			
				var[0] = lexicon.getVariable(Inference.getVariableName(eventId, Inference.O_ID, "event"));
				coef[0] = -1;
				
				var[1] = lexicon.getVariable(Inference.getVariableName(entityId, Inference.O_ID, "entity"));
				coef[1] = 1;
				
				constraints.add(new ILPConstraint(var, coef, 0,
						ILPConstraint.GREATER_THAN));
				
				//each event has at least one entity
				vars[counter] = lexicon.getVariable(Inference.getVariableName(entityId, Inference.E_ID, "entity"));
				coefs[counter] = 1;
			    counter++;
			}
			/*constraints.add(new ILPConstraint(vars, coefs, 0,
					ILPConstraint.GREATER_THAN));*/
			
		}
		
		//entity -> event
		/*for (Integer entityId : entityToEvent.keySet()) {
			int[] var = new int[2];
			double[] coef = new double[2];

			// the first var in the constraint says that eventId -> A. The other vars
			// say that other slots map to B

			var[0] = lexicon.getVariable(Inference.getVariableName(entityId, Inference.E_ID, "entity"));
			coef[0] = -1;
			
			for(Integer eventId : entityToEvent.get(entityId)){
				var[1] = lexicon.getVariable(Inference.getVariableName(eventId, Inference.E_ID, "event"));
				coef[1] = 1;
				constraints.add(new ILPConstraint(var, coef, 0,
						ILPConstraint.GREATER_THAN));
			}
		}*/

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
