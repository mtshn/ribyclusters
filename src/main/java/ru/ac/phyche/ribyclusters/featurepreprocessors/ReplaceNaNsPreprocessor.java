package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

/**
 * 
 * This preprocessor replaces NaN values with -1 values. No features are
 * excluded. File format. The first line: "PREPROCESSOR
 * ReplaceNaNsPreprocessor". The second line is the same as the first one:
 * "PREPROCESSOR ReplaceNaNsPreprocessor". The third one contains one integer
 * number: number of features. The fourth line contains space-separated names of
 * the features. The 5th line: "END".
 *
 */
public class ReplaceNaNsPreprocessor extends FeaturesPreprocessor {

	private String[] names = null;

	@Override
	public void train(FeaturesGenerator features, ChemDataset data) {
		features.precompute(data);
		names = features.getNames();
	}

	@Override
	public void save(FileWriter filewriter) throws IOException {
		filewriter.write("PREPROCESSOR ReplaceNaNsPreprocessor\n");
		filewriter.write("PREPROCESSOR ReplaceNaNsPreprocessor\n");
		filewriter.write(names.length + "\n");
		for (int i = 0; i < names.length; i++) {
			filewriter.write(names[i] + " ");
		}
		filewriter.write("\n");
		filewriter.write("END\n");
	}

	@Override
	public void load(BufferedReader filereader) throws IOException {
		String s = filereader.readLine();
		while (s.trim().equals("")) {
			s = filereader.readLine();
		}
		if (!s.trim().split("\\s+")[0].equals("PREPROCESSOR")) {
			throw (new IOException("Wrong file format! Word PREPROCESSOR is expected"));
		}
		if (!s.trim().split("\\s+")[1].equals("ReplaceNaNsPreprocessor")) {
			throw (new IOException("Wrong file format!"));
		}
		while (s.trim().split("\\s+")[0].equals("PREPROCESSOR")
				&& s.trim().split("\\s+")[1].equals("ReplaceNaNsPreprocessor")) {
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
		}
		if (s.trim().equals("0")) {
			names = new String[] {};
		} else {
			int n = Integer.parseInt(s.trim());
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
			names = s.trim().split("\\s+");
			if (names.length != n) {
				throw (new IOException("Wrong file format!"));
			}
		}
		s = filereader.readLine();
		while (s.trim().equals("")) {
			s = filereader.readLine();
		}
		if (!s.trim().equals("END")) {
			throw (new IOException("Wrong file format!"));
		}
	}

	@Override
	public float[] preprocess(float[] input) {
		float[] outp = new float[input.length];
		if (input.length != names.length) {
			throw (new RuntimeException("Wrong length of array! Preprocessing failed"));
		}
		for (int i = 0; i < outp.length; i++) {
			outp[i] = Float.isNaN(input[i]) ? -1 : input[i];
		}
		return outp;
	}

	@Override
	public String[] featureNames() {
		return names;
	}

}
