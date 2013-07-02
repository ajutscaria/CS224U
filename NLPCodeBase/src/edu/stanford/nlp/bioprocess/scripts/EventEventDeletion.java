package edu.stanford.nlp.bioprocess.scripts;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Set;
import java.util.TreeSet;

import edu.stanford.nlp.io.IOUtils;


public class EventEventDeletion {

	public static void main(String[] args) throws IOException {

//		String randomProcessesFile  = args[0]; // contains names of files we want to delete relations for.
		File processesDir = new File(args[1]); //contains directory with the annotations.
		String outDir = args[2];

		//Set<String> randomProcessNames = extractProcessName(randomProcessesFile);
		for(File f: processesDir.listFiles()) {
		//	if(randomProcessNames.contains(f.getName())) {
				PrintWriter writer = IOUtils.getPrintWriter(outDir+f.getName());
				for(String line: IOUtils.readLines(f)) {
					if(f.getName().endsWith(".txt"))
						writer.println(line);
					else if(line.startsWith("T")) {
						String[] tokens = line.split("\t");
						int index = Integer.parseInt(tokens[0].substring(1));
						writer.println(line);
						if(tokens[1].startsWith("Event"))
							writer.println("E"+index+"\tEvent:T"+index);
						if(tokens[1].startsWith("Static-Event"))
							writer.println("E"+index+"\tStatic-Event:T"+index);
					}
				}
				writer.close();
		//	}
		}
	}

	private static Set<String> extractProcessName(String randomProcessesFile) throws IOException {		
		Set<String> res = new TreeSet<String>();
		for(String line: IOUtils.readLines(randomProcessesFile)) {
			int pIndex = line.indexOf('p');
			if(pIndex==-1)
				throw new RuntimeException("p character not where supposed to be");
			res.add(line.substring(pIndex,line.lastIndexOf('.'))+".ann");
			res.add(line.substring(pIndex,line.lastIndexOf('.'))+".txt");

		}
		return res;
	}

}
