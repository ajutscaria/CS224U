package edu.stanford.nlp.bioprocess.ilp.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


//import cs224n.util.CounterMap;
import LBJ2.infer.ILPSolver;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.infer.ilp.AbstractILPInference;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.BioDatum;
import edu.stanford.nlp.trees.Tree;

public class Inference extends AbstractILPInference<ExampleStructure> {

	public static final String[] eventLabels = { "E", "O"};
	public static final String[] entityLabels = { "E", "O"};
	public static String[] relationLabels = {};

	// the id for A & B in the above list. This can be done better
	public static final int E_ID = 0;
	public final static int O_ID = 1;

	private List<ILPConstraintGenerator> constraints;
	private List<BioDatum> eventPredicted;
	private List<BioDatum> entityPredicted;
	private List<BioDatum> relationPredicted;
	public HashMap<Integer, HashSet<Integer>> eventToEntity = new HashMap<Integer, HashSet<Integer>>();	
	public HashMap<Integer, HashSet<Integer>> entityToEvent = new HashMap<Integer, HashSet<Integer>>();
	public HashMap<Integer, HashSet<Integer>> entityChildren = new HashMap<Integer, HashSet<Integer>>();
	//private CounterMap<String,String> wordToTagCounters = new CounterMap<String, String>();
	//wordToTagCounters.incrementCount(word, tag, 1.0);
	
	public Inference(List<BioDatum> eventPredicted, List<BioDatum> entityPredicted, List<BioDatum> relationPredicted,
			ILPSolverFactory solverFactory,
			boolean debug) {
		super(solverFactory, debug);
		System.out.println("ILP inference initialized");
		System.out.println(eventPredicted.size());
		System.out.println(entityPredicted.size());
		System.out.println(relationPredicted.size());
		this.eventPredicted = eventPredicted;
		this.entityPredicted = entityPredicted;
		this.relationPredicted = relationPredicted;
		relationLabels = new String[relationPredicted.get(0).rankRelation.size()];
		relationLabels[0] = "NONE";
		int count = 1;
		for(String key: relationPredicted.get(0).rankRelation.keySet()){
			if(key.equals("NONE"))continue;
			//System.out.println("Relation: "+ key);
			relationLabels[count] = key;
			count++;
		}
	    //System.out.println("Number of relations: "+relationLabels.length);
		//System.out.println("eventpredicted: "+eventPredicted.size());
		HashSet<Integer> temp = new HashSet<Integer>();
		/*for(int j=0; j<entityPredicted.size();j++){
			int whichevent = entityPredicted.get(j).eventId;
			if(eventToEntity.containsKey(whichevent)){
				//System.out.println(j);
				temp = eventToEntity.get(whichevent);
				temp.add(j);
				eventToEntity.put(whichevent, temp);
			}else{
				temp = new HashSet<Integer>();
				temp.add(j);
				//System.out.println(j);
				eventToEntity.put(whichevent, temp);
			}
			temp = new HashSet<Integer>();
			temp.add(whichevent);
			entityToEvent.put(j, temp);
		}*/
        
		for(int i=0; i<eventPredicted.size(); i++){
			//System.out.println("Event "+i + ":");
			for(int j=0; j<entityPredicted.size();j++){
				if(!eventPredicted.get(i).eventNode.equals(entityPredicted.get(j).eventNode) || 
						!eventPredicted.get(i).getSentence().equals(entityPredicted.get(j).getSentence()))
					continue;
				else{
					if(eventToEntity.containsKey(i)){
						//System.out.println(j);
						temp = eventToEntity.get(i);
						temp.add(j);
						eventToEntity.put(i, temp);
					}else{
						temp = new HashSet<Integer>();
						temp.add(j);
						//System.out.println(j);
						eventToEntity.put(i, temp);
					}
					if(entityToEvent.containsKey(j)){
						temp = entityToEvent.get(j);
						temp.add(i);
						entityToEvent.put(j, temp);
					}else{
						temp = new HashSet<Integer>();
						temp.add(i);
						entityToEvent.put(j, temp);
					}
				}
			}
		}
		temp = new HashSet<Integer>();
		for(int key:eventToEntity.keySet()){
			for(int value:eventToEntity.get(key)){
				temp.add(value);
			}
		}
		System.out.println("Number of entities belonging to events: "+temp.size());
		for(int i=0; i<entityPredicted.size();i++){
			List<Tree> children_original =  entityPredicted.get(i).entityNode.getChildrenAsList();
			HashSet<Tree> children = new HashSet<Tree>();
			children.addAll(children_original);
			//System.out.println("parent: "+i);
			for(int j=0; j<entityPredicted.size();j++){
				if(j==i || !entityPredicted.get(i).getSentence().equals(entityPredicted.get(j).getSentence())
						|| !children.contains(entityPredicted.get(j).entityNode))
						continue;
				else{
					if(entityChildren.containsKey(i)){
						//System.out.println(j);
						temp = entityChildren.get(i);
						temp.add(j);
						entityChildren.put(i, temp);
					}else{
						//System.out.println(j);
						temp = new HashSet<Integer>();
						temp.add(j);
						entityChildren.put(i, temp);
					}
				}		
			}
		}
		System.out.println("eventToEntity size: " + eventToEntity.size());
		System.out.println("entityToEvent size: " + entityToEvent.size());
		System.out.println("entityChildren size: " + entityChildren.size());
		constraints = new ArrayList<ILPConstraintGenerator>();
		/*constraints.add(new UniqueLabelConstraint());
		constraints.add(new ValidAConstraintGenerator());
		constraints.add(new EntityChildrenConstraintGenerator());*/
	}

	@Override
	protected void addConstraints(ILPSolver solver,
			InferenceVariableLexManager lexicon) {
		System.out.println("Start adding constraints");
		UniqueLabelConstraint unique = new UniqueLabelConstraint();
		ValidAConstraintGenerator valid = new ValidAConstraintGenerator();
		EntityChildrenConstraintGenerator children = new EntityChildrenConstraintGenerator();
		RelationEventConstraintGenerator relation = new RelationEventConstraintGenerator();
		ExampleInput eventinput = new ExampleInput("event", eventPredicted.size(), eventLabels.length, eventPredicted);
		ExampleInput entityinput = new ExampleInput("entity", entityPredicted.size(), entityLabels.length, entityPredicted);
		ExampleInput relationinput = new ExampleInput("relation", relationPredicted.size(), relationLabels.length, relationPredicted);
		
		System.out.println("Start adding unique constraints");
		for (ILPConstraint constraint : unique.getILPConstraints(eventinput, lexicon))
			this.addConstraint(solver, constraint);
		for (ILPConstraint constraint : unique.getILPConstraints(entityinput, lexicon))
			this.addConstraint(solver, constraint);
		for (ILPConstraint constraint : unique.getILPConstraints(relationinput, lexicon))
			this.addConstraint(solver, constraint);
		System.out.println("finish adding unique constraints");
		
		System.out.println("Start adding entity to event constraints");
		ExampleInput entitydualevent= new ExampleInput("", entityToEvent.size(), entityLabels.length, eventToEntity, entityToEvent);
		for (ILPConstraint constraint : valid.getILPConstraints(entitydualevent, lexicon))
			this.addConstraint(solver, constraint);
		System.out.println("finish adding event to entity and entity to event constraints");
		
		System.out.println("Start adding entity children constraints");
		ExampleInput entitychild= new ExampleInput("entity", entityChildren.size(), entityLabels.length, entityChildren);
		for (ILPConstraint constraint : children.getILPConstraints(entitychild, lexicon))
			this.addConstraint(solver, constraint);
		System.out.println("finish adding entity children constraints");
		
		System.out.println("Start adding event-event relation constraints");
		for (ILPConstraint constraint : relation.getILPConstraints(relationinput, lexicon))
			this.addConstraint(solver, constraint);
		System.out.println("finish adding event-event relation constraints");
		
		System.out.println("done adding constraints");
	}

	@Override
	protected void addVariables(ILPSolver solver,
			InferenceVariableLexManager lexicon) {
		// each label can take one of the five values

		System.out.println("start adding variables");
		for (int eventId = 0; eventId < eventPredicted.size(); eventId++) {
				double score = Math.log(eventPredicted.get(eventId).getEventProbability());
			    //System.out.println("Score of E for event "+ eventId + ": " + score);

				// create a boolean variable with this score
				int var = solver.addBooleanVariable(score);
				String varName = getVariableName(eventId, E_ID, "event"); // 0: E, 1: O
				lexicon.addVariable(varName, var);
				
				//** should use just one var!
				score = Math.log(1-eventPredicted.get(eventId).getEventProbability());
				//System.out.println("Score of O for event "+ eventId + ": " + score);
				var = solver.addBooleanVariable(score);
				varName = getVariableName(eventId, O_ID, "event"); // 0: E, 1: O
				lexicon.addVariable(varName, var);
		}
		
		for (int entityId = 0; entityId < entityPredicted.size(); entityId++) {
			
			//for (int labelId = 0; labelId < entityLabels.length; labelId++) {

				// get the variable objective coefficient for the variable to be added
				double score = Math.log(entityPredicted.get(entityId).getProbability());
				//System.out.println("Score of E for entity "+ entityId + ": " + entityPredicted.get(entityId).getProbability());
				// create a boolean variable with this score
				int var = solver.addBooleanVariable(score);
				String varName = getVariableName(entityId, E_ID, "entity");
				lexicon.addVariable(varName, var);
				
				score = Math.log(1-entityPredicted.get(entityId).getProbability());
				// create a boolean variable with this score
				var = solver.addBooleanVariable(score);
				varName = getVariableName(entityId, O_ID, "entity");
				lexicon.addVariable(varName, var);
			//}	
		}
        for (int Id = 0; Id < relationPredicted.size(); Id++) {
			
			for (int labelId = 0; labelId < relationLabels.length; labelId++) {

				// get the variable objective coefficient for the variable to be added
				double score = relationPredicted.get(Id).getRelationProb(relationLabels[labelId]);
				//System.out.println("Score of "+relationLabels[labelId]+ " for relation "+ Id + ": " + score);
				// create a boolean variable with this score
				int var = solver.addBooleanVariable(score);
				String varName = getVariableName(Id, labelId, "relation");
				lexicon.addVariable(varName, var);

			}			
		}
		System.out.println("done adding variables");
	}

	public static String getVariableName(int eventId, int labelId, String type) {
		return type + eventId + ",label" + labelId;
	}
	

	@Override
	protected ExampleStructure getOutput(ILPSolver solver,
			InferenceVariableLexManager lexicon) throws Exception {
 
		StringBuilder original = new StringBuilder();
		for(int i=0;i < eventPredicted.size(); i++){
			original.append(eventPredicted.get(i).guessLabel);
		}
		int count = 0;
		int EtoO = 0;
		int event = 0;
		System.out.println("getoutput");
		for (int eventId = 0; eventId < eventPredicted.size(); eventId++) {
			for (int labelId = 0; labelId < eventLabels.length; labelId++){
				String varName = getVariableName(eventId, labelId, "event");
				int var = lexicon.getVariable(varName);

				if (solver.getBooleanValue(var)) {
					if(eventLabels[labelId].equals("E")){
						event++;
					}
					if(!eventLabels[labelId].equals(eventPredicted.get(eventId).guessLabel)){
						count++;
						//System.out.println("Event Different: original - "+eventPredicted.get(eventId).guessLabel + ", after - "+eventLabels[labelId]);
						if(eventPredicted.get(eventId).guessLabel.equals("E") && eventLabels[labelId].equals("O"))
							EtoO++;
					}
					eventPredicted.get(eventId).setPredictedLabel(eventLabels[labelId]);
					break;
				}
			}
		}
		System.out.println("predicted event:"+event);
		System.out.println("Different events:" + count+ ", E->O: "+ EtoO);
		count = 0;
		EtoO = 0;
		int entity = 0;
		for (int entityId = 0; entityId < entityPredicted.size(); entityId++) {
            for (int labelId = 0; labelId < entityLabels.length; labelId++){
            	String varName = getVariableName(entityId, labelId, "entity");
				int var = lexicon.getVariable(varName);

				if (solver.getBooleanValue(var)) {
					if(entityLabels[labelId].equals("E")){
						entity++;
					}
					if(!entityLabels[labelId].equals(entityPredicted.get(entityId).guessLabel)){
						//System.out.println("Entity Different: original - "+entityPredicted.get(entityId).guessLabel + ", after - "+entityLabels[labelId]);
					    count++;
					    if(entityPredicted.get(entityId).guessLabel.equals("E") && entityLabels[labelId].equals("O"))
							EtoO++;
					}
					entityPredicted.get(entityId).setPredictedLabel(entityLabels[labelId]);
					break;
				}
			}
		}
		System.out.println("entities predicted: "+entity);
		System.out.println("Different entities:" + count+ ", E->O: "+ EtoO);
		int notNone = 0;
		int changedtoNone = 0;
		for (int relationId = 0; relationId < relationPredicted.size(); relationId++) {
			for (int labelId = 0; labelId < relationLabels.length; labelId++){
				String varName = getVariableName(relationId, labelId, "relation");
				int var = lexicon.getVariable(varName);

				if (solver.getBooleanValue(var)) {
					if(!relationLabels[labelId].equals(relationPredicted.get(relationId).guessLabel)){
						//System.out.println("True: "+ relationPredicted.get(relationId).label +", Original predicted: " + relationPredicted.get(relationId).guessLabel + ", ILP: "+relationLabels[labelId]);
						count++;
						if(relationLabels[labelId].equals("NONE"))
							changedtoNone++;
					}
					if(!relationLabels[labelId].equals("NONE")){
						//System.out.println("True: "+ relationPredicted.get(relationId).label +", Original predicted: " + relationPredicted.get(relationId).guessLabel + ", ILP: "+relationLabels[labelId]);
						//System.out.println("Relation event1: "+ relationPredicted.get(relationId).event1.getTreeNode().toString()
						//		+", event2: "+ relationPredicted.get(relationId).event2.getTreeNode().toString());
						//System.out.println("True event1: "+ eventPredicted.get(relationPredicted.get(relationId).event1_index).eventNode.toString()
						//		+", event2: "+ eventPredicted.get(relationPredicted.get(relationId).event2_index).eventNode.toString());
						notNone++;
						//System.out.println("\n");
					}
					relationPredicted.get(relationId).setPredictedLabel(relationLabels[labelId]);
					break;
				}
			}
		}
		System.out.println("Different relations:" + count+", changed into NONE: "+changedtoNone);
		System.out.println("Predicted as not none:" + notNone);
		ExampleInput dummy = new ExampleInput("", 0, 0, new ArrayList<BioDatum>());
		String [] label = new String[1];
		return new ExampleStructure(dummy, label);
	}
}



/*
public class Inference extends AbstractILPInference<ExampleStructure> {

	public static final String[] validLabels = { "A", "B", "C", "D", "E" };

	// the id for A & B in the above list. This can be done better
	public static final int A_ID = 0;
	public final static int B_ID = 1;

	private List<ILPConstraintGenerator> constraints;
	private ExampleInput input;

	public Inference(ExampleInput input, ILPSolverFactory solverFactory,
			boolean debug) {
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

				// get the variable objective coefficient for the variable to be added
				double score = getLabelScore(slotId, labelId);

				// create a boolean variable with this score
				int var = solver.addBooleanVariable(score);

				// the next two steps will help to remember this variable in the lexicon
				// for later use.
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
	protected ExampleStructure getOutput(ILPSolver solver,
			InferenceVariableLexManager lexicon) throws Exception {

		String[] labels = new String[input.slots];
		for (int slotId = 0; slotId < input.slots; slotId++) {
			for (int labelId = 0; labelId < validLabels.length; labelId++) {
				String varName = getVariableName(slotId, labelId);

				int var = lexicon.getVariable(varName);

				if (solver.getBooleanValue(var)) {
					labels[slotId] = validLabels[labelId];
					break;
				}
			}
		}

		return new ExampleStructure(input, labels);
	}
}
*/