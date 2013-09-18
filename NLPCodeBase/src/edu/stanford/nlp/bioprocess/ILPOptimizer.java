
package edu.stanford.nlp.bioprocess;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;
import gurobi.*;


public class ILPOptimizer {
	static int MaxVariables = 0, MaxConstraints = 0;
	private HashMap<String, Double> weights;
	int numEvents, numLabels, NONEIndex = 0;
	int cotemporalEventIndex, sameEventIndex, previousEventIndex, causesIndex, causedIndex, enablesIndex, 
					enabledIndex, nextEventIndex, superEventIndex, subEventIndex;
	static int closeToOne, allVariables;
	private double alpha1 = 0;
	private double alpha2 = 0;
	private double alpha3 = 0;
	private double alpha4 = 0;
	private double alpha5 = 0;
	private double alpha6 = 0;
	private double alpha7 = 0;
	private boolean includeConnectedComponentConstraint = false,  
					includeSameEventHardConstraint = false, includePreviousHardConstraint = false, 
					includeSameEventContradictionsHardConstraint = false;
	List<String> labels;
	IntCounter<String> goldTripleCounts;	
	Triple<String, String, String> causesCausesCotemporalTriple = new Triple<String, String, String>("Causes", "CotemporalEvent", "Causes");
	HashMap<String, GRBVar> X_ij = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> X_ijr = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> Y_ij = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> Phi_ij = new HashMap<String, GRBVar>();
	HashMap<String, Integer> eventTypeIndex = new HashMap<String, Integer>();
	
	//Indicator variable for Cotemporal event soft constraint penalize
	HashMap<String, GRBVar> Z1_ijk = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> Z2_ijk = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> Z3_ijk = new HashMap<String, GRBVar>();
	
	//Indicator variable for Same event soft constraint with rewards
	HashMap<String, GRBVar> A_ijk = new HashMap<String, GRBVar>();
	
	//Indicator variable for Same event soft constraint with penalty
	HashMap<String, GRBVar> B1_ijk = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> B2_ijk = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> B3_ijk = new HashMap<String, GRBVar>();
	
	//Indicator variable for Past event hard constraints
	HashMap<String, GRBVar> C_ijPast = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> C_ijNotPast = new HashMap<String, GRBVar>();
	
	//Indicator variable for Present event hard constraints
	HashMap<String, GRBVar> C_ijPresent = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> C_ijNotPresent = new HashMap<String, GRBVar>();
	
	//Indicator variable for Future event hard constraints
	HashMap<String, GRBVar> C_ijFuture = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> C_ijNotFuture = new HashMap<String, GRBVar>();
	
	//Indicator variable for Sub event hard constraints
	HashMap<String, GRBVar> C_ijSub = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> C_ijNotSub = new HashMap<String, GRBVar>();
	
	//Indicator variable for Super event hard constraints
	HashMap<String, GRBVar> C_ijSuper = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> C_ijNotSuper = new HashMap<String, GRBVar>();
	
	//Indicator variable for Cotemporal triad closure soft constraint with rewards
	HashMap<String, GRBVar> D_ijk = new HashMap<String, GRBVar>();

	//Indicator variable for Causes->Cotemporal->Causes event triad closure soft constraint with rewards
	//HashMap<String, GRBVar> E1_ijk = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> E2_ijk = new HashMap<String, GRBVar>();
	//HashMap<String, GRBVar> E3_ijk = new HashMap<String, GRBVar>();
	HashMap<Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>>, GRBVar> E_ijk = new HashMap<Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>>, GRBVar>();
	HashMap<Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>>, GRBVar> F_ijk = new HashMap<Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>>, GRBVar>();
	
	//Indicator variable for chain constraint : soft
	HashMap<String, GRBVar> G_i = new HashMap<String, GRBVar>();
	
	GRBEnv	env;
	GRBModel  model;
	private final int topK = 3;
	
	public ILPOptimizer(HashMap<String, Double> weights,
			int numEvents, List<String> labels, boolean connectedComponentIn, boolean sameEventIn, boolean previousEventIn,
			boolean sameEventContradictionIn,
			double alpha1In, double alpha2In, double alpha3In, double alpha4In, double alpha5In, double alpha6In, double alpha7In) {
		this.setWeights(weights);
		this.numEvents = numEvents;
		this.labels = labels;
		this.numLabels = labels.size();
		this.includeConnectedComponentConstraint = connectedComponentIn;
		this.includeSameEventHardConstraint = sameEventIn;
		this.includePreviousHardConstraint = previousEventIn;
		this.includeSameEventContradictionsHardConstraint = sameEventContradictionIn;
		this.alpha1 = alpha1In;
		this.alpha2 = alpha2In;
		this.alpha3 = alpha3In;
		this.alpha4 = alpha4In;
		this.alpha5 = alpha5In;
		this.alpha6 = alpha6In;
		this.alpha7 = alpha7In;

		for(String eventType:ArgumentRelation.getEventRelations()) {
			eventTypeIndex.put(eventType, labels.indexOf(eventType));
		}
		
		try {
			this.cotemporalEventIndex = eventTypeIndex.get("CotemporalEvent");
			this.sameEventIndex = eventTypeIndex.get("SameEvent");
			this.causedIndex = eventTypeIndex.get("Caused");
			this.causesIndex = eventTypeIndex.get("Causes");
			this.enablesIndex = eventTypeIndex.get("Enables");
			this.previousEventIndex = eventTypeIndex.get("PreviousEvent");
			this.nextEventIndex = eventTypeIndex.get("NextEvent");
			this.enabledIndex = eventTypeIndex.get("Enabled");
			this.superEventIndex = eventTypeIndex.get("SuperEvent");
			this.subEventIndex = eventTypeIndex.get("SubEvent");
			
			goldTripleCounts = (IntCounter<String>)Utils.readObject("GoldTriplesCount.ser");
			/*
			IntCounter<String> backup = (IntCounter<String>)goldTripleCounts.clone();
			
			goldTripleCounts = new IntCounter<String>();
			for(String key:backup.keySet()) {
				if(backup.getCount(key) > 5) {
					goldTripleCounts.setCount(key, backup.getCount(key));
				}
			}
			*/
			//LogInfo.logs("\nGold triple count:" + goldTripleCounts.size());
			//LogInfo.logs(goldTripleCounts.keySet());
			
			env = new GRBEnv("1.log");
			model = new GRBModel(env);
			
			//Creating X_ijr variables(undirected) i<j, r \in R
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					for(int r=0;r<numLabels; r++) {
						X_ijr.put(String.format("%d,%d,%d", i, j, r), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, r)));
					}
				}
			}
			
			//Creating X_ij variables(undirected) i<j
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					X_ij.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijFuture.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijPast.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijPresent.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijSub.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijSuper.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					/*
					C_ijNotFuture.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijNotPast.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijNotPresent.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijNotSub.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					C_ijNotSuper.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
					*/
				}
			}
			
			//Creating Y_ij and Phi_ij variables(directed) i<j
			for(int i=0; i<numEvents; i++) {
				for(int j=0; j<numEvents; j++) {
					if(j!=i) {
						Y_ij.put(String.format("%d,%d", i, j), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
							String.format("%d,%d",i, j)));
						Phi_ij.put(String.format("%d,%d", i, j), model.addVar(0.0, numEvents, 0.0, GRB.INTEGER, 
								String.format("%d,%d",i, j)));
					}
				}
			}
			
			//Creating Z_ij undirected for i<j
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++)
					for(int k=j+1; k<numEvents; k++){
						Z1_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
						Z2_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
						Z3_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
				}
			}
			
			//Creating A_ij and B_ij undirected for i<j
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++)
					for(int k=j+1; k<numEvents; k++){
						A_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
						B1_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
						B2_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
						B3_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
						D_ijk.put(String.format("%d,%d,%d", i, j, k), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d",i, j, k)));
				}
			}
			
			//Creating F_ijk and G_i for counts
			for(int i=0; i<numEvents; i++) {
				G_i.put(String.format("%d", i), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
						String.format("%d", i)));
				for(int j=i+1; j<numEvents; j++) {
					for(int k=j+1; k<numEvents; k++) {
						/*for(String key:goldTripleCounts.keySet()) {
							String[] triple = key.split(",");
							//int tripleCount = 1;
							//LogInfo.logs("Equivalent triples:"+key);
							//LogInfo.logs(Utils.getEquivalentTriples(new Triple<String, String, String>(triple[0], triple[1], triple[2])));
							for(Triple<String, String, String> expansion:Utils.getEquivalentTriples(new Triple<String, String, String>(triple[0], triple[1], triple[2]))) {
									F_ijk.put(new Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>>(new Triple<Integer, Integer, Integer>(i, j, k), new Triple<String, String, String>(expansion.first(), expansion.second(), expansion.third())), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
										String.format("%d,%d,%d,%s,%s,%s",i, j, k, expansion.first(), expansion.second(), expansion.third())));
								//tripleCount++;
							}
						}*/
						for(Triple<String, String, String> expansion:Utils.getEquivalentTriples(causesCausesCotemporalTriple)) {
							E_ijk.put(new Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>>(new Triple<Integer, Integer, Integer>(i, j, k), new Triple<String, String, String>(expansion.first(), expansion.second(), expansion.third())), model.addVar(0.0, 1.0, 0.0, GRB.BINARY, 
								String.format("%d,%d,%d,%s,%s,%s",i, j, k, expansion.first(), expansion.second(), expansion.third())));
						}
					}
				}
			}
			//for(Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>> ppp: F_ijk.keySet()) {
				//LogInfo.logs(ppp);
			//}
			model.update();
			
			//Set objective - maximize score
			GRBLinExpr expr = new GRBLinExpr();
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					for(int r=0;r<numLabels; r++) {
						//System.out.println(i + "," + j + "," + r);
						expr.addTerm(weights.get(String.format("%d,%d,%d", i, j, r)), 
										X_ijr.get(String.format("%d,%d,%d", i, j, r)));
					}
					for(int k=j+1; k<numEvents; k++) {
						expr.addTerm(-1 * getAlpha1(), Z1_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(-1 * getAlpha1(), Z2_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(-1 * getAlpha1(), Z3_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(getAlpha2(), A_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(-1 * getAlpha3(), B1_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(-1 * getAlpha3(), B2_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(-1 * getAlpha3(), B3_ijk.get(String.format("%d,%d,%d", i, j, k)));
						expr.addTerm(getAlpha4(), D_ijk.get(String.format("%d,%d,%d", i, j, k)));
					}
				}
				expr.addTerm(-1 * getAlpha7(), G_i.get(String.format("%d", i)));
			}
			
			//Adding counts to objective
			for(Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>> key:E_ijk.keySet()) {
				expr.addTerm(getAlpha5(), E_ijk.get(key));
			}
			/*
			for(Pair<Triple<Integer, Integer, Integer>, Triple<String, String, String>> key:F_ijk.keySet()) {
				Triple<String, String, String> baseTriple = Utils.getEquivalentBaseTriple(key.second());
				String tripleString = String.format("%s,%s,%s", baseTriple.first(), baseTriple.second(), baseTriple.third());
				//LogInfo.logs("Key in F_ijk  :" + tripleString);
				//LogInfo.logs("Base          :" + Utils.getEquivalentBaseTriple(key.second()));
				expr.addTerm(getAlpha6() * Math.log(goldTripleCounts.getCount(tripleString)), F_ijk.get(key));
				//if(!goldTripleCounts.containsKey(tripleString)){
				//	LogInfo.logs("BIGGG PROBLEMMM");
				//}
				//LogInfo.logs(goldTripleCounts.getCount(tripleString)+ ":" + F_ijk.get(key));
			}*/
			
		    model.setObjective(expr, GRB.MAXIMIZE);
		}
		catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
	}
	/**
	 * The label NONE always comes at 0th index.
	 * @param weights - 
	 * @param numEvents
	 * @param numLabels
	 */
	public HashMap<Pair<Integer,Integer>, Integer> OptimizeEventRelation() {
		HashMap<Pair<Integer,Integer>, Integer> best = new HashMap<Pair<Integer,Integer>, Integer>();
		if(!includeConnectedComponentConstraint && !includePreviousHardConstraint && !includePreviousHardConstraint
				&& alpha1 == 0 && alpha2 == 0 && alpha3 == 0 && alpha4 == 0 && alpha5 == 0 && alpha6 == 0)
			return best; 
		//LogInfo.logs(String.format("Current values %b, %b, %b, %b, %f, %f, %f, %f, %f, %f, %f\n", isIncludeConnectedComponentConstraint(),
		//		isIncludeSameEventHardConstraint(),
		//		isIncludePreviousHardConstraint(), includeSameEventContradictionsHardConstraint,
		//		getAlpha1(), getAlpha2(), getAlpha3(), getAlpha4(), getAlpha5(), getAlpha6(), getAlpha7()));
		try
		{		
			
		    //Set connected component constraints
			if(includeConnectedComponentConstraint) {
				addConnectedComponentConstraints();
			}
		      
		    //Constraint for SameEvent triad closure 
			if(includeSameEventHardConstraint) {
				addSameEventTriadClosureHardConstraints();
			}
	    	
	    	//Constraint for PreviousEvent
			if(includePreviousHardConstraint) {
				addPreviousEventHardConstraints();
			}
		    
		    //Hard constraint for same event contradictions
			if(includeSameEventContradictionsHardConstraint) {
				addSameEventContradictionHardConstarints();
			}
			
		    //Soft constraint for cotemporal penalize
		    addCotemporalSoftConstraintsPenalize();
		    
		    //Soft constraint for cotemporal reward
		    addCotemporalSoftConstraintsReward();
		    
		    //Soft constraint for same event with reward
		    addSameEventSoftConstraintsReward();
		    
		    //Soft constraint for same event with penalty
		    addSameEventSoftConstraintsPenalize();
		    
		    //Soft constraint for Causes->Cotemporal->Causes with reward
		    addCausesSoftConstraintsReward();
		    
		    //Soft constraint for favoring triples in GOLD.
		    //addConstraintforGOLDTriplesReward();
		    
		    addLinearChainSoftConstraintPenalize();
		    
		    model.update();
		    
		    //Count number of variables
		    if(model.getVars().length > MaxVariables)
		    	MaxVariables = model.getVars().length;
		    
		    //Count number of constraints
		    if(model.getConstrs() != null) {
			    if(model.getConstrs().length > MaxConstraints)
			    	MaxConstraints = model.getConstrs().length;
		    }
		    
		    //Optimize the model
		    model.optimize();
		    
		    
		    /*model.computeIIS();
		    
		    for (GRBConstr c : model.getConstrs())
		    {
		        if (c.get(GRB.IntAttr.IISConstr) > 0)
		        {
		            System.out.println(c.get(GRB.StringAttr.ConstrName));
		        }
		    }                

		    // Print the names of all of the variables in the IIS set.
		    for (GRBVar v : model.getVars()) {
		        if (v.get(GRB.IntAttr.IISLB) > 0 || v.get(GRB.IntAttr.IISUB) > 0)
		        {
		            System.out.println(v.get(GRB.StringAttr.VarName));
		        }
		    }*/
		    // Dispose of model and environment
		    
		    /*for(GRBVar x:X_ijr.values()) {
		      //LogInfo.logs(x.get(GRB.StringAttr.VarName)
                     // + " " +x.get(GRB.DoubleAttr.X));
		      //if(!x.get(GRB.StringAttr.VarName).endsWith(",0") && x.get(GRB.DoubleAttr.X) == 1) {
		      if(x.get(GRB.DoubleAttr.X) == 1) {
		    		String splits[] = x.get(GRB.StringAttr.VarName).split(",");
		    		best.put(new Pair<Integer, Integer>(Integer.parseInt(splits[0]), Integer.parseInt(splits[1])), Integer.parseInt(splits[2]));
		    	}
		    }*/
		    for(int i = 0; i < numEvents; i++) {
		    	for(int j = i+1; j < numEvents; j++) {
		    		best.put(new Pair<Integer, Integer>(i, j), getBestRelation(i, j));
		    	}
		    }
		    
		    model.dispose();
		    env.dispose();
		}
		catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
		//System.out.println(best);
		return best;
	}
	
	private void addLinearChainSoftConstraintPenalize() {
		try{
			for(int j=0; j<numEvents; j++) {
				GRBLinExpr chainConstraint1 = new GRBLinExpr();

				for(int i=0;i<numEvents;i++) {
					if(i < j) {
						chainConstraint1.addTerm(1.0, X_ij.get(String.format("%d,%d", i, j)));
					}
					else if(i > j) {
						chainConstraint1.addTerm(1.0, X_ij.get(String.format("%d,%d", j, i)));
					}
		    	}
				chainConstraint1.addTerm(-1 * (numEvents - 3), G_i.get(((Integer)j).toString()));
				model.addConstr(chainConstraint1, GRB.LESS_EQUAL, 2, String.format("c50_%d", j));
				
				GRBLinExpr chainConstraint2 = new GRBLinExpr();
				for(int i=0;i<numEvents;i++) {
					if(i < j) {
						chainConstraint2.addTerm(-1.0, X_ij.get(String.format("%d,%d", i, j)));
					}
					else if(i > j) {
						chainConstraint2.addTerm(-1.0, X_ij.get(String.format("%d,%d", j, i)));
					}
		    	}
				chainConstraint2.addTerm(3, G_i.get(((Integer)j).toString()));
				

				model.addConstr(chainConstraint2, GRB.LESS_EQUAL, 0, String.format("c50_%d", j));
			}
		}
		catch(GRBException ex) {
			ex.printStackTrace();
		}
	}
	
	private int getBestRelation(int i, int j) {
		HashMap<String,Double> map = new HashMap<String,Double>();
        ValueComparator bvc =  new ValueComparator(map);
        TreeMap<String,Double> sorted_map = new TreeMap<String,Double>(bvc);
		
		for(String label:labels){
			try {
				map.put(((Integer)labels.indexOf(label)).toString(), X_ijr.get(String.format("%d,%d,%d", i, j, labels.indexOf(label))).get(GRB.DoubleAttr.X));
			} catch (GRBException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//System.out.println(map);
		sorted_map.putAll(map);
		
		for(String key: sorted_map.keySet()) {
			//System.out.println(map.get(key));
			if(1- map.get(key) <= 0.0001) 
				closeToOne++;
			allVariables++;
			//System.out.println("Returning " + key);
			return Integer.parseInt(key);
		}
		
		return -1;
	}
	
	private void addConstraintforGOLDTriplesReward() {
		try{
			 for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			for(String goldTriple:goldTripleCounts.keySet()) {
		    				String[] triple = goldTriple.split(",");
							List<Triple<String, String, String>> equivalentTriples = Utils.getEquivalentTriples(new Triple<String, String, String>(triple[0], triple[1], triple[2]));
							//int tripleCount = 1;
			    			for(Triple<String, String, String> equivalentTriple : equivalentTriples) {
			    				//LogInfo.logs(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple));
			    				//LogInfo.logs(F_ijk.get(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple)));
								GRBLinExpr sameEventConstraint = new GRBLinExpr();
								if(Utils.isInTopK(weights, labels, i, j, equivalentTriple.first, topK) && 
										Utils.isInTopK(weights, labels, j, k, equivalentTriple.second, topK) &&
										Utils.isInTopK(weights, labels, i, k, equivalentTriple.third, topK)) {
					    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, eventTypeIndex.get(equivalentTriple.first))));
					    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, eventTypeIndex.get(equivalentTriple.second))));
					    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, eventTypeIndex.get(equivalentTriple.third))));
					    			sameEventConstraint.addTerm(3.0, F_ijk.get(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple)));
									model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 0.0, String.format("c18_%d,%d,%d", i,j,k));
									//System.out.println(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple));
									
									sameEventConstraint = new GRBLinExpr();
									sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, eventTypeIndex.get(equivalentTriple.first))));
					    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, eventTypeIndex.get(equivalentTriple.second))));
					    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, eventTypeIndex.get(equivalentTriple.third))));
					    			sameEventConstraint.addTerm(-1.0, F_ijk.get(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple)));
									model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c19_%d,%d,%d", i,j,k));
									//tripleCount++;
								}
			    			}
		    			}
		    		}
		    	}
			 }
		 }
		 catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	     }
	}
	private void addSameEventSoftConstraintsPenalize() {
		try{
			 for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			GRBLinExpr sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, B1_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c24_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
						sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(3.0, B1_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c25_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, B2_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c26_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
						sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(3.0, B2_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c27_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, B3_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c28_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
						sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(3.0, B3_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c29_%d,%d,%d", i,j,k));
		    		}
		    	}
			 }
		 }
		 catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	     }
	}
	
	private void addSameEventSoftConstraintsReward() {
		try{
			 for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			GRBLinExpr sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(3.0, A_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 0.0, String.format("c18_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
						sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, A_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c19_%d,%d,%d", i,j,k));
		    		}
		    	}
			 }
		 }
		 catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	     }
	}
	
	private void addCausesSoftConstraintsReward() {
		try{
			//(Causes,CotemporalEvent,Causes), (Caused,Causes,CotemporalEvent), (CotemporalEvent,Caused,Caused), 
			 for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			List<Triple<String, String, String>> equivalentTriples = Utils.getEquivalentTriples(causesCausesCotemporalTriple);
		    			for(Triple<String, String, String> equivalentTriple : equivalentTriples) {
		    				GRBLinExpr sameEventConstraint = new GRBLinExpr();
			    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, eventTypeIndex.get(equivalentTriple.first))));
			    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, eventTypeIndex.get(equivalentTriple.second))));
			    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, eventTypeIndex.get(equivalentTriple.third))));
			    			sameEventConstraint.addTerm(3.0, E_ijk.get(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple)));
							model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 0.0, String.format("c18_%d,%d,%d", i,j,k));
							
							sameEventConstraint = new GRBLinExpr();
							sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, eventTypeIndex.get(equivalentTriple.first))));
			    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, eventTypeIndex.get(equivalentTriple.second))));
			    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, eventTypeIndex.get(equivalentTriple.third))));
			    			sameEventConstraint.addTerm(-1.0, E_ijk.get(new Pair<Triple<Integer,Integer,Integer>, Triple<String,String,String>>(new Triple<Integer, Integer, Integer>(i, j, k), equivalentTriple)));
							model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c19_%d,%d,%d", i,j,k));
		    			}
		    		}
		    	}
			 }
		 }
		 catch (GRBException e) {
			  LogInfo.logs(e);
		      LogInfo.logs("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	     }
	}
	
	private void addCotemporalSoftConstraintsReward() {
		try{
			 for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			GRBLinExpr sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			sameEventConstraint.addTerm(3.0, D_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 0.0, String.format("c18_%d,%d,%d", i,j,k));
						
						sameEventConstraint = new GRBLinExpr();
						sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, D_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c19_%d,%d,%d", i,j,k));
		    		}
		    	}
			 }
		 }
		 catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	     }
	}
	
	private void addCotemporalSoftConstraintsPenalize() {
		 try{
			 for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			GRBLinExpr cotemporalEventConstraint = new GRBLinExpr();
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, Z1_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(cotemporalEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c16_%d,%d,%d", i,j,k));
						
						cotemporalEventConstraint = new GRBLinExpr();
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			cotemporalEventConstraint.addTerm(3.0, Z1_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(cotemporalEventConstraint, GRB.LESS_EQUAL, 5.0, String.format("c17_%d,%d,%d", i,j,k));
						
						cotemporalEventConstraint = new GRBLinExpr();
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, Z2_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(cotemporalEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c16_%d,%d,%d", i,j,k));
						
						cotemporalEventConstraint = new GRBLinExpr();
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			cotemporalEventConstraint.addTerm(3.0, Z2_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(cotemporalEventConstraint, GRB.LESS_EQUAL, 5.0, String.format("c17_%d,%d,%d", i,j,k));
						
						cotemporalEventConstraint = new GRBLinExpr();
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, Z3_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(cotemporalEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c16_%d,%d,%d", i,j,k));
						
						cotemporalEventConstraint = new GRBLinExpr();
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, cotemporalEventIndex)));
		    			cotemporalEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			cotemporalEventConstraint.addTerm(3.0, Z3_ijk.get(String.format("%d,%d,%d", i, j, k)));
						model.addConstr(cotemporalEventConstraint, GRB.LESS_EQUAL, 5.0, String.format("c17_%d,%d,%d", i,j,k));
		    		}
		    	}
			 }
		 }
		 catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
	}
	
	private void addSameEventContradictionHardConstarints() {
		try {
			//There are 5 categories: Past, Future, Sub, Super and Present.
			//Defining variables
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					GRBLinExpr sumRelationsPair = new GRBLinExpr();
					
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, previousEventIndex)));
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, causesIndex)));
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, enablesIndex)));
					sumRelationsPair.addTerm(-1.0, C_ijPast.get(String.format("%d,%d", i, j)));
					model.addConstr(sumRelationsPair, GRB.EQUAL, 0.0, String.format("c2_%d,%d", i,j));
					
					sumRelationsPair = new GRBLinExpr();
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, nextEventIndex)));
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, causedIndex)));
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, enabledIndex)));
					sumRelationsPair.addTerm(-1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
					model.addConstr(sumRelationsPair, GRB.EQUAL, 0.0, String.format("c2_%d,%d", i,j));
					
					sumRelationsPair = new GRBLinExpr();
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, superEventIndex)));
					sumRelationsPair.addTerm(-1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
					model.addConstr(sumRelationsPair, GRB.EQUAL, 0.0, String.format("c2_%d,%d", i,j));
					
					sumRelationsPair = new GRBLinExpr();
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, subEventIndex)));
					sumRelationsPair.addTerm(-1.0, C_ijSub.get(String.format("%d,%d", i, j)));
					model.addConstr(sumRelationsPair, GRB.EQUAL, 0.0, String.format("c2_%d,%d", i,j));
					
					sumRelationsPair = new GRBLinExpr();
					sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, cotemporalEventIndex)));
					sumRelationsPair.addTerm(-1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
					model.addConstr(sumRelationsPair, GRB.EQUAL, 0.0, String.format("c2_%d,%d", i,j));
				}
			}
			for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			//For conflict between past and future in case of same event
		    			GRBLinExpr previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
		    			//For conflict between past and sub event in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
		    			//For conflict between past and super event in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
		    			//For conflict between past and present in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
		    			//For conflict between future and sub event in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						//For conflict between future and super event in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						//For conflict between future and present in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijFuture.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPast.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						//For conflict between sub and super event in case of same event
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						//For conflict between sub and present in case of same event
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						//For conflict between super and present in case of same event
		    			previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, j)));
		    			previousEventConstraint.addTerm(1.0, C_ijSuper.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
						previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			previousEventConstraint.addTerm(1.0, C_ijSub.get(String.format("%d,%d", j, k)));
		    			previousEventConstraint.addTerm(1.0, C_ijPresent.get(String.format("%d,%d", i, k)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 2.0, String.format("c10_%d,%d,%d", i,j,k));
		    		}
		    	}
		    }
		}
		catch(GRBException e) {
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
                      e.getMessage());
		}
	}
	
	private void addPreviousEventHardConstraints() {
		/*
	     * Constraint for ensuring a--Prev-->b, b--Prev-->c, a--NONE-->c. (along with the six equivalent configurations)
	     */
	    /*(PreviousEvent,PreviousEvent,NONE)
	    (NONE,NextEvent,PreviousEvent)
	    (NextEvent,NONE,PreviousEvent)
	    (PreviousEvent,NONE,NextEvent)
	    (NONE,PreviousEvent,NextEvent)
	    (NextEvent,NextEvent,NONE)*/
		try{
	    	int nextEventIndex = labels.indexOf("NextEvent"), previousEventIndex = labels.indexOf("PreviousEvent");
		    for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			GRBLinExpr previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, previousEventIndex)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, previousEventIndex)));
		    			previousEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, NONEIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c10_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, nextEventIndex)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, previousEventIndex)));
		    			previousEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, NONEIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c11_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, nextEventIndex)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, previousEventIndex)));
		    			previousEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, NONEIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c12_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, previousEventIndex)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, nextEventIndex)));
		    			previousEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, NONEIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c13_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, previousEventIndex)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, nextEventIndex)));
		    			previousEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, NONEIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c14_%d,%d,%d", i,j,k));
						
						previousEventConstraint = new GRBLinExpr();
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, nextEventIndex)));
		    			previousEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, nextEventIndex)));
		    			previousEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, NONEIndex)));
						model.addConstr(previousEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c15_%d,%d,%d", i,j,k));
		    		}
		    	}
		    }
		}
		catch (GRBException e) {
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
	}
	
	private void addSameEventTriadClosureHardConstraints() {
		try {
			int sameEventIndex = labels.indexOf("SameEvent");
		    for(int i=0; i<numEvents; i++) {
		    	for(int j=i+1;j<numEvents;j++) {
		    		for(int k=j+1;k<numEvents;k++) {
		    			GRBLinExpr sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c7_%d,%d,%d", i,j,k));
						//LogInfo.logs(String.format("SCC ij, jk, ik %s-%s,%s-%s,%s-%s", i,j, j,k, i,k));
						
						sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c8_%d,%d,%d", i,j,k));
						//LogInfo.logs(String.format("SCC ik, jk, ij %s-%s,%s-%s,%s-%s", i,k, j,k, i,j));
						
						sameEventConstraint = new GRBLinExpr();
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, sameEventIndex)));
		    			sameEventConstraint.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, k, sameEventIndex)));
		    			sameEventConstraint.addTerm(-1.0, X_ijr.get(String.format("%d,%d,%d", j, k, sameEventIndex)));
						model.addConstr(sameEventConstraint, GRB.LESS_EQUAL, 1.0, String.format("c9_%d,%d,%d", i,j,k));
						//LogInfo.logs(String.format("SCC ij, ik, jk %s-%s,%s-%s,%s-%s", i,j, i,k, j,k));
		    		}
		    	}
		    }
		}
		catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
	}
	
	private void addConnectedComponentConstraints() {
		try {
			//For each unordered pair \sum{r \in R} X_ijr = 1
		    for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					GRBLinExpr oneRelationForPair = new GRBLinExpr();
					for(int r=0;r<numLabels; r++) {
						oneRelationForPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, r)));
					}
					model.addConstr(oneRelationForPair, GRB.EQUAL, 1.0, String.format("c1_%d,%d", i,j));
				}
			}
		    //For unordererd each pair X_ij = \sum{r \in R'} X_ijr
		    for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					GRBLinExpr sumRelationsPair = new GRBLinExpr();
					for(int r=1;r<numLabels; r++) {
						sumRelationsPair.addTerm(1.0, X_ijr.get(String.format("%d,%d,%d", i, j, r)));
					}
					sumRelationsPair.addTerm(-1.0, X_ij.get(String.format("%d,%d", i, j)));
					model.addConstr(sumRelationsPair, GRB.EQUAL, 0.0, String.format("c2_%d,%d", i,j));
				}
			}
		    
		    //For each ordered pair X_ij = \sum{r \in R'} X_ijr
		    for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					GRBLinExpr directedIndicator1 = new GRBLinExpr();
					directedIndicator1.addTerm(1.0, Y_ij.get(String.format("%d,%d", i, j)));
					//directedIndicator1.addTerm(1.0, Y_ij.get(String.format("%d,%d", j, i)));
					directedIndicator1.addTerm(-1.0, X_ij.get(String.format("%d,%d", i, j)));
					model.addConstr(directedIndicator1, GRB.LESS_EQUAL, 0.0, String.format("c3_%d,%d", i,j));
					
					GRBLinExpr directedIndicator2 = new GRBLinExpr();
					directedIndicator2.addTerm(1.0, Y_ij.get(String.format("%d,%d", j, i)));
					directedIndicator2.addTerm(-1.0, X_ij.get(String.format("%d,%d", i, j)));
					model.addConstr(directedIndicator2, GRB.LESS_EQUAL, 0.0, String.format("c3_%d,%d", j,i));
				}
			}
		    
		    //Incoming edges degree. 0th index is root and has 0 in-degree.
		    for(int j=0; j<numEvents; j++) {
		    	GRBLinExpr inDegree = new GRBLinExpr();
				for(int i=0; i<numEvents; i++) {
					if(j!=i) {
						inDegree.addTerm(1.0, Y_ij.get(String.format("%d,%d", i, j)));
					}
				}
				if(j==0) {
					model.addConstr(inDegree, GRB.EQUAL, 0.0, String.format("c4_%d", j));
				}
				else{
					model.addConstr(inDegree, GRB.EQUAL, 1.0, String.format("c4_%d", j));
				}
			}
		     
		    //Flow. 0th index is root and has flow n-1.
		    for(int j=0; j<numEvents; j++) {
		    	GRBLinExpr flow = new GRBLinExpr();
		    	if(j==0) {
		    		for(int k=0; k<numEvents; k++) {
						if(k!=j) {
							flow.addTerm(1.0, Phi_ij.get(String.format("%d,%d", j, k)));
						}
					}
					model.addConstr(flow, GRB.EQUAL, numEvents-1, String.format("c5_%d", j));
				}
		    	else {
					for(int i=0; i<numEvents; i++) {
						if(j!=i) {
							flow.addTerm(1.0, Phi_ij.get(String.format("%d,%d", i, j)));
						}
					}
					for(int k=0; k<numEvents; k++) {
						if(j!= 0 && k!=j) {
							flow.addTerm(-1.0, Phi_ij.get(String.format("%d,%d", j, k)));
						}
					}
					model.addConstr(flow, GRB.EQUAL, 1.0, String.format("c5_%d", j));
		    	}
			}
		    
		    //Limit
		    for(int i=0; i<numEvents; i++) {
		    	for(int j=0; j<numEvents; j++) {
		    		if(i!=j) {
			    		GRBLinExpr limit = new GRBLinExpr();
			    		limit.addTerm(1.0, Phi_ij.get(String.format("%d,%d", i, j)));
			    		limit.addTerm(-1 * numEvents, Y_ij.get(String.format("%d,%d", i, j)));
			    		model.addConstr(limit, GRB.LESS_EQUAL, 0.0, String.format("c6_%d,%d", i,j));
		    		}
		    	}
		    }
		}
		catch (GRBException e) {
			  System.out.println(e);
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
	}
	
	public static void Optimize() {
		try {
		      GRBEnv    env   = new GRBEnv("mip1.log");
		      GRBModel  model = new GRBModel(env);

		      // Create variables

		      GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "x");
		      GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "y");
		      GRBVar z = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "z");

		      // Integrate new variables

		      model.update();

		      // Set objective: maximize x + y + 2 z

		      GRBLinExpr expr = new GRBLinExpr();
		      expr.addTerm(1.0, x); expr.addTerm(1.0, y); expr.addTerm(2.0, z);
		      model.setObjective(expr, GRB.MAXIMIZE);

		      // Add constraint: x + 2 y + 3 z <= 4

		      expr = new GRBLinExpr();
		      expr.addTerm(1.0, x); expr.addTerm(2.0, y); expr.addTerm(3.0, z);
		      model.addConstr(expr, GRB.LESS_EQUAL, 4.0, "c0");

		      // Add constraint: x + y >= 1

		      expr = new GRBLinExpr();
		      expr.addTerm(1.0, x); expr.addTerm(1.0, y);
		      model.addConstr(expr, GRB.GREATER_EQUAL, 1.0, "c1");

		      // Optimize model

		      model.optimize();

		      System.out.println(x.get(GRB.StringAttr.VarName)
		                         + " " +x.get(GRB.DoubleAttr.X));
		      System.out.println(y.get(GRB.StringAttr.VarName)
		                         + " " +y.get(GRB.DoubleAttr.X));
		      System.out.println(z.get(GRB.StringAttr.VarName)
		                         + " " +z.get(GRB.DoubleAttr.X));

		      System.out.println("Obj: " + model.get(GRB.DoubleAttr.ObjVal));

		      // Dispose of model and environment

		      model.dispose();
		      env.dispose();

	    } 
		catch (GRBException e) {
		      System.out.println("Error code: " + e.getErrorCode() + ". " +
		                         e.getMessage());
	    }
  	}
	public double getAlpha1() {
		return alpha1;
	}
	public void setAlpha1(double alpha1In) {
		alpha1 = alpha1In;
	}
	public double getAlpha2() {
		return alpha2;
	}
	public void setAlpha2(double alpha2In) {
		alpha2 = alpha2In;
	}
	public double getAlpha3() {
		return alpha3;
	}
	public void setAlpha3(double alpha3In) {
		alpha3 = alpha3In;
	}
	public boolean isIncludeConnectedComponentConstraint() {
		return includeConnectedComponentConstraint;
	}
	public void setIncludeConnectedComponentConstraint(
			boolean includeConnectedComponentConstraintIn) {
		includeConnectedComponentConstraint = includeConnectedComponentConstraintIn;
	}
	public boolean isIncludePreviousHardConstraint() {
		return includePreviousHardConstraint;
	}
	public void setIncludePreviousHardConstraint(
			boolean includePreviousHardConstraintIn) {
		includePreviousHardConstraint = includePreviousHardConstraintIn;
	}
	public boolean isIncludeSameEventHardConstraint() {
		return includeSameEventHardConstraint;
	}
	public void setIncludeSameEventHardConstraint(
			boolean includeSameEventHardConstraintIn) {
		includeSameEventHardConstraint = includeSameEventHardConstraintIn;
	}
	HashMap<String, Double> getWeights() {
		return weights;
	}
	void setWeights(HashMap<String, Double> weights) {
		this.weights = weights;
	}
	public double getAlpha4() {
		return alpha4;
	}
	public void setAlpha4(double alpha4) {
		this.alpha4 = alpha4;
	}
	public double getAlpha5() {
		return alpha5;
	}
	public void setAlpha5(double alpha5) {
		this.alpha5 = alpha5;
	}
	public double getAlpha6() {
		return alpha6;
	}
	public void setAlpha6(double alpha6) {
		this.alpha6 = alpha6;
	}
	private double getAlpha7() {
		return alpha7;
	}
	private void setAlpha7(double alpha7) {
		this.alpha7 = alpha7;
	}
}

