package edu.stanford.nlp.bioprocess.scripts;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessFormatReader;
import edu.stanford.nlp.bioprocess.EventMention;
import edu.stanford.nlp.bioprocess.Example;
import edu.stanford.nlp.bioprocess.Utils;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import fig.basic.LogInfo;

public class ComputeAgreement {

	public static int tp=0,fp=0,fn=0,tn=0;
	public static double[][] kappa;
	private static Map<String,Integer> relToIndex = new HashMap<String, Integer>();
	private static boolean collapsed = true;
	//private static String[] processIds = {"p31","p33","p35","p36","p37","p38","p39","p42","p46","p4"};
	//private static String[] processIds = {"p47","p5","p53","p55","p56","p57","p59","p6","p62","p64","p67","p7","p75","p76","p77",
		//"p81","p83","p86","p91"};
	private static String[] processIds = {"p50","p147","p87","p139","p65","p25","p2","p3","p90",
		"p28","p142","p68","p45","p51","p32","p61","p16","p19","p84","p74","p82","p43","p96","p148","p49",
		"p99","p94","p136","p149","p58"};

	public static void main(String[] args) throws IOException {

		int i = 0;
		if(collapsed) {
			relToIndex.put("CotemporalEvent", i++);
			relToIndex.put("NextEvent", i++);
			relToIndex.put("PreviousEvent", i++);
			relToIndex.put("SameEvent", i++);
			relToIndex.put("SuperEvent", i++);
			relToIndex.put("SubEvent", i++);
			relToIndex.put("NONE", i++);
		}
		else {
			relToIndex.put("CotemporalEvent", i++);
			relToIndex.put("NextEvent", i++);
			relToIndex.put("PreviousEvent", i++);
			relToIndex.put("SameEvent", i++);
			relToIndex.put("SuperEvent", i++);
			relToIndex.put("SubEvent", i++);
			relToIndex.put("Causes", i++);
			relToIndex.put("Caused", i++);
			relToIndex.put("Enables", i++);
			relToIndex.put("Enabled", i++);
			relToIndex.put("NONE", i++);
		}

		if(collapsed) {
			kappa = new double[7][7];
		}
		else {
			kappa = new double[11][11];
		}
		String brittanyTemp = "brittanyTemp/";
		String justinTemp = "justinTemp/";
		createTemp(brittanyTemp,args[0]);
		createTemp(justinTemp,args[1]);

		computeAgreement(brittanyTemp,justinTemp);		
	}

	private static void createTemp(String tempDir, String fullDir) {
		fig.basic.Utils.systemHard("mkdir -p " + tempDir);
		fig.basic.Utils.systemHard("rm -f " + tempDir+"*");
		for(String pId: processIds) {
			fig.basic.Utils.systemHard("cp " + fullDir+pId+".txt"+" "+tempDir+pId+".txt");
			fig.basic.Utils.systemHard("cp " + fullDir+pId+".ann"+" "+tempDir+pId+".ann");
		}
	}

	private static void computeAgreement(String brittany, String justin) throws IOException {

		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		StanfordCoreNLP processor = new StanfordCoreNLP(props, false);
		BioProcessFormatReader processReader = new BioProcessFormatReader();
		processReader.setProcessor(processor);

		Map<String,Example> brittanyAnnotations = toMap(processReader.parseFolder(brittany));
		Map<String,Example> justinAnnotations = toMap(processReader.parseFolder(justin));
		for(String brittanyKey: brittanyAnnotations.keySet()) {
			Example brittanyExample = brittanyAnnotations.get(brittanyKey);
			Example justinExample = justinAnnotations.get(brittanyKey);
			if(justinExample==null) throw new RuntimeException("process missing from justin annotations: " + brittanyKey);
			handleExample(brittanyExample,justinExample);
		}
		double recall = (double) tp / (tp + fn);
		double precision = (double) tp / (tp + fp);
		double f1 = (2*recall*precision)/(precision+recall);
		System.out.println("recall\t"+recall+"\tprecision\t"+precision+"\tf1\t"+f1);
		ComputeKappa.findKappa(kappa);
	}

	private static void handleExample(Example brittanyExample,
			Example justinExample) {

		List<EventMention> brittanyMentions = brittanyExample.gold.get(EventMentionsAnnotation.class);
		List<EventMention> justinMentions = justinExample.gold.get(EventMentionsAnnotation.class);
		if(brittanyMentions.size()!=justinMentions.size()) 
			//throw new RuntimeException("Number of event mentions is not identical");
			return;
		for(int i = 0; i < brittanyMentions.size()-1; ++i) {
			for(int j = i+1; j < brittanyMentions.size(); ++j) {
				String brittanyRelation = collapse(Utils.getEventEventRelation(brittanyExample.gold, brittanyMentions.get(i).getTreeNode(),
						brittanyMentions.get(j).getTreeNode()).toString());
				String justinRelation = collapse(Utils.getEventEventRelation(justinExample.gold, justinMentions.get(i).getTreeNode(),
						justinMentions.get(j).getTreeNode()).toString());
				handleRelation(brittanyRelation,justinRelation,brittanyExample.id);
			}
		}
	}

	private static void handleRelation(String brittanyRelation,
			String justinRelation, String exampleId) {

		Integer brittanyIndex = relToIndex.get(brittanyRelation);
		Integer justinIndex = relToIndex.get(justinRelation);
		if(brittanyIndex==null || justinIndex==null)
			throw new RuntimeException("Unknown relation: " + brittanyRelation + " " + justinRelation);
		kappa[brittanyIndex][justinIndex]++;

		if(brittanyRelation.equals(justinRelation)) {
			if(brittanyRelation.equals("NONE"))
				tn++;
			else
				tp++;
		}
		else {
			if(justinRelation.equals("NONE")) { 
				fp++;
				LogInfo.logs(exampleId+" "+ "brittany: " + brittanyRelation + " justin: " + justinRelation);
			}
			else if(brittanyRelation.equals("NONE")) {
				fn++;
				LogInfo.logs(exampleId+" "+ "brittany: " + brittanyRelation + " justin: " + justinRelation);
			}
			else {
				fp++;
				fn++;
				LogInfo.logs(exampleId+" "+ "brittany: " + brittanyRelation + " justin: " + justinRelation);
			}
		}
	}

	private static String collapse(String relation) {

		if(collapsed) {
			if(relation.equals("Causes"))
				return "PreviousEvent";
			if(relation.equals("Caused"))
				return "NextEvent";
			if(relation.equals("Enables"))
				return "PreviousEvent";
			if(relation.equals("Enabled"))
				return "NextEvent";
			return relation;
		}
		else {
			return relation;
		}
	}

	public static Map<String,Example> toMap(List<Example> examples) {
		Map<String,Example> res = new HashMap<String, Example>();
		for(Example example: examples) {
			res.put(example.id, example);
		}
		return res;
	}

}
