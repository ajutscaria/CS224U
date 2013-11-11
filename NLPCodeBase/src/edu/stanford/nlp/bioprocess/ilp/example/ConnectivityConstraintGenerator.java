package edu.stanford.nlp.bioprocess.ilp.example;

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

public class ConnectivityConstraintGenerator extends ILPConstraintGenerator {

	public ConnectivityConstraintGenerator() {
		super("Zij, Yij, PHIij", false);
	}

	@Override
	public List<ILPConstraint> getILPConstraints(IInstance all, InferenceVariableLexManager lexicon) {

		ExampleInput input = (ExampleInput)all;
		List<BioDatum> relationPredicted = input.data;
		
		List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
		int slotlength = input.labels;
		System.out.println("Inside connectivity constraint");
        System.out.println("relationpredicted size: "+relationPredicted.size());
        HashMap<String, List<Integer>> processToEvent = Inference.processToEvent;
		   
        for (int Id = 0; Id < relationPredicted.size(); Id++) {
        	int event1 = relationPredicted.get(Id).event1_index;
			int event2 = relationPredicted.get(Id).event2_index;
			
			// Yij = SUM(Yijr), r != NONE
			int[] vars = new int[slotlength];
			double[] coefficients = new double[slotlength];
			for (int labelId = 1; labelId < slotlength; labelId++) {
				vars[labelId] = lexicon.getVariable(Inference.getVariableName(event1, event2, labelId, "relation"));
				coefficients[labelId] = 1.0;
			}
			vars[0] = lexicon.getVariable(Inference.getVariableName(event1, event2, "edge", "connectivity")); //Yij
			coefficients[0] = -1;
			constraints.add(new ILPConstraint(vars, coefficients, 0, ILPConstraint.EQUAL));
			
			
			//equation 2
			int[] var = new int[2];
			double[] coef = new double[2];
			// Zij <= Yij
			var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2, "edge", "connectivity")); //Yij
			coef[0] = -1;
			var[1] = lexicon.getVariable(Inference.getVariableName(event1, event2, "aux", "connectivity")); //Zij
			coef[1] = 1;
			constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.LESS_THAN));
			// Zji <= Yij
			var[1] = lexicon.getVariable(Inference.getVariableName(event2, event1, "aux", "connectivity")); //Zji
			coef[1] = 1;
			constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.LESS_THAN));
        }
        
        
        for(String process : processToEvent.keySet()){
        	List<Integer> events = processToEvent.get(process);
        	int size = events.size()-1;
        	int[] var = new int[size];
			double[] coef = new double[size];
        	int root = events.get(0);
        	// equation 3
        	for(int i = 0; i < events.size();i++){
        		int counter = 0;
        		int event1 = events.get(i);
        		for(int j=0; j<events.size();j++){
        			int event2 = events.get(j);
        			if(i==j)continue;
        			var[counter] = lexicon.getVariable(Inference.getVariableName(event2, event1, "aux", "connectivity"));
          		    coef[counter] = 1;
        			counter++;
        		}
        		if(i == 0) //root
        		   constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.EQUAL));
        		else
        		   constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.EQUAL));
        	}
        	
        	//equation 4
        	/*int counter = 0;
        	for(int i = 1; i < events.size();i++){
        		int event2 = events.get(i);
        		var[counter] = lexicon.getVariable(Inference.getVariableName(root, event2, "flow", "connectivity"));
      		    coef[counter] = 1;
      		    counter++;
        	}
        	constraints.add(new ILPConstraint(var, coef, size, ILPConstraint.EQUAL));
        	
        	//equation 5
        	int doublesize = 2*size;
        	var = new int[doublesize];
			coef = new double[doublesize];
			for(int j = 1; j < events.size();j++){
				int eventj = events.get(j);
				counter = 0;
				for(int i=0; i<events.size();i++){
					if(i==j)continue;
					int eventi = events.get(i);
					var[counter] = lexicon.getVariable(Inference.getVariableName(eventi, eventj, "flow", "connectivity"));
	      		    coef[counter] = 1;
	      		    counter++;
				}
				for(int k=0; k<events.size();k++){
					if(j==k)continue;
					int eventk = events.get(k);
					var[counter] = lexicon.getVariable(Inference.getVariableName(eventj, eventk, "flow", "connectivity"));
	      		    coef[counter] = -1;
	      		    counter++;
				}
				constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.EQUAL));
        	}
        	
			//equation 6
			int n = events.size();
			var = new int[2];
			coef = new double[2];
			for(int i=0; i<events.size();i++){
				int eventi = events.get(i);
				for(int j=0; j<events.size();j++){
					if(i==j)continue;
					int eventj = events.get(j);
					var[0] = lexicon.getVariable(Inference.getVariableName(eventi, eventj, "flow", "connectivity")); //PHIij
					coef[0] = 1;
					var[1] = lexicon.getVariable(Inference.getVariableName(eventi, eventj, "aux", "connectivity")); //Zij
					coef[1] = -n;
					constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.LESS_THAN));
				}
			}*/
			
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

