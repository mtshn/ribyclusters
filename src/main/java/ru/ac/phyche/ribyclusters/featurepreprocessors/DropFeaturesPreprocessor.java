package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * 
 * This preprocessor drops (excludes) some features from the initial feature set
 * and creates the reduced feature set. See sub-classes for details. See
 * superclass for more detailed using information. File format. The first line:
 * "PREPROCESSOR DropFeaturesPreprocessor". The second line is the same as the
 * first one: "PREPROCESSOR DropFeaturesPreprocessor". The third one contains
 * one integer number: number of features that should be retained (number of
 * features after preprocessing). The fourth line contains space-separated names
 * of the retained features (after preprocessing). The 5th line contains (space
 * separated) indices of features (in the non-preprocessed array) to be
 * excluded. The 6th line: "END". The file format is the same for all
 * sub-classes.
 *
 */
public abstract class DropFeaturesPreprocessor extends FeaturesPreprocessor {

	private String[] names = null;
	private int[] featuresToDrop = new int[] {};

	/**
	 * 
	 * @return featureNames()
	 */
	public String[] getNames() {
		return names;
	}

	/**
	 * Set names of features. The length of the names array must correspond to
	 * number of features after preprocessing.
	 * 
	 * @param names feature names
	 */
	public void setNames(String[] names) {
		this.names = names;
	}

	/**
	 * 
	 * @return indices of features that should be excluded from the initial feature
	 *         set. Indices in the initial array (non-preprocessed) are given.
	 */
	public int[] getFeaturesToDrop() {
		return featuresToDrop;
	}

	/**
	 * 
	 * @param featuresToDrop indices of features that should be excluded from the
	 *                       initial feature set. Indices in the initial array
	 *                       (non-preprocessed) are given.
	 */
	public void setFeaturesToDrop(int[] featuresToDrop) {
		this.featuresToDrop = featuresToDrop;
	}

	@Override
	public void save(FileWriter filewriter) throws IOException {
		filewriter.write("PREPROCESSOR DropFeaturesPreprocessor\n");
		filewriter.write("PREPROCESSOR DropFeaturesPreprocessor\n");
		filewriter.write(names.length + "\n");
		for (int i = 0; i < names.length; i++) {
			filewriter.write(names[i] + " ");
		}
		filewriter.write("\n");
		for (int i = 0; i < featuresToDrop.length; i++) {
			filewriter.write(featuresToDrop[i] + " ");
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
		if (!s.trim().split("\\s+")[1].equals("DropFeaturesPreprocessor")) {
			throw (new IOException("Wrong file format!"));
		}
		while (s.trim().split("\\s+")[0].equals("PREPROCESSOR")
				&& s.trim().split("\\s+")[1].equals("DropFeaturesPreprocessor")) {
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
			String[] ints = s.trim().split("\\s+");
			featuresToDrop = new int[ints.length];
			for (int i = 0; i < featuresToDrop.length; i++) {
				featuresToDrop[i] = Integer.parseInt(ints[i]);
			}
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
		}
		if (!s.trim().equals("END")) {
			throw (new IOException("Wrong file format!"));
		}
	}

	@Override
	public float[] preprocess(float[] input) {
		if (input.length - featuresToDrop.length != names.length) {
			throw (new RuntimeException("Wrong length of array! Preprocessing failed"));
		}
		if (featuresToDrop.length == 0) {
			return input.clone();
		}
		float[] result = new float[input.length - featuresToDrop.length];
		int i = 0;
		int k = 0;
		for (int j = 0; j < input.length; j++) {
			if (k < featuresToDrop.length && (featuresToDrop[k] == j)) {
				k++;
			} else {
				result[i] = input[j];
				i++;
			}
		}

		return result;
	}

	@Override
	public String[] featureNames() {
		return names;
	}

}
