package edu.stanford.nlp.bioprocess;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;
import gurobi.*;


public class ILPOptimizer {
	static int MaxVariables = 0, MaxConstraints = 0;
	HashMap<String, Double> weights;
	int numEvents, numLabels, NONEIndex = 0;;
	List<String> labels;
	
	HashMap<String, GRBVar> X_ij = new HashMap<String, GRBVar>();
	
	HashMap<String, GRBVar> X_ijr = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> Y_ij = new HashMap<String, GRBVar>();
	HashMap<String, GRBVar> Phi_ij = new HashMap<String, GRBVar>();
	GRBEnv	env;
	GRBModel  model;
	
	public ILPOptimizer(HashMap<String, Double> weights,
			int numEvents, List<String> labels) {
		this.weights = weights;
		this.numEvents = numEvents;
		this.labels = labels;
		this.numLabels = labels.size();
		try {
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
			
			model.update();
			
			//Set objective - maximize score
			GRBLinExpr expr = new GRBLinExpr();
			for(int i=0; i<numEvents; i++) {
				for(int j=i+1; j<numEvents; j++) {
					for(int r=0;r<numLabels; r++) {
						expr.addTerm(weights.get(String.format("%d,%d,%d", i, j, r)), 
										X_ijr.get(String.format("%d,%d,%d", i, j, r)));
					}
				}
			}
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
		try
		{		
			
		    //Set connected component constraints
		    addConnectedComponentConstraints();
		      
		    //Constraint for SameEvent triad closure
	    	addSameEventTriadClosureConstraints();
	    	
	    	//Constraint for PreviousEvent
	    	addPreviousEventConstraints();
		    
		    model.update();
		    
		    //Count number of variables
		    if(model.getVars().length > MaxVariables)
		    	MaxVariables = model.getVars().length;
		    
		    //Count number of constraints
		    if(model.getConstrs().length > MaxConstraints)
		    	MaxConstraints = model.getConstrs().length;
		    
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
		    
		    for(GRBVar x:X_ijr.values()) {
		      //LogInfo.logs(x.get(GRB.StringAttr.VarName)
                     // + " " +x.get(GRB.DoubleAttr.X));
		      //if(!x.get(GRB.StringAttr.VarName).endsWith(",0") && x.get(GRB.DoubleAttr.X) == 1) {
		      if(x.get(GRB.DoubleAttr.X) == 1) {
		    		String splits[] = x.get(GRB.StringAttr.VarName).split(",");
		    		best.put(new Pair<Integer, Integer>(Integer.parseInt(splits[0]), Integer.parseInt(splits[1])), Integer.parseInt(splits[2]));
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
		System.out.println(best);
		return best;
	}
	
	private void addPreviousEventConstraints() {
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
	private void addSameEventTriadClosureConstraints() {
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
}

