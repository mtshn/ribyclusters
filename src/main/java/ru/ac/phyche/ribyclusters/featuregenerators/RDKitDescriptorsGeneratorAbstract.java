package ru.ac.phyche.ribyclusters.featuregenerators;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import org.apache.commons.lang3.SystemUtils;

/**
 * 
 * Abstract class for multi-threaded computation of molecular descriptors and
 * fingerprints using RDKit. Python executable (with RDkit and other required
 * dependencies installed) should be located in "./python/bin/python3" (for
 * Linux and Mac os x). Windows is not supported now. It is full path to the
 * executable: python3 is a file. If you have other location of python - please
 * fix the python location in the source code of this class and recompile the
 * project. Usually we use the portable edition of Python from this repository:
 * https://github.com/indygreg/python-build-standalone/ The rdkit-pypi, pandas
 * packages are required. Read doc for the pythonScriptFileToRunRDKit() method.
 * The descriptor-generating python scripts are described there.
 *
 */
public abstract class RDKitDescriptorsGeneratorAbstract extends FeaturesGenerator {
	@Override
	public void precompute(HashSet<String> smilesStrings) {
		HashSet<String> smilesStrings1 = new HashSet<String>();
		for (String s : smilesStrings) {
			if (!this.precomputedForMol(s)) {
				smilesStrings1.add(s);
			}
		}
		if (smilesStrings1.size() != 0) {
			int nProcs = Runtime.getRuntime().availableProcessors();
			nProcs = Math.min(nProcs, smilesStrings1.size());
			String[] smiles = smilesStrings1.toArray(new String[smilesStrings1.size()]);
			int subsetSize = smiles.length / nProcs;
			for (int i = 0; i < nProcs; i++) {
				int min = i * subsetSize;
				int max = (i + 1) * subsetSize;
				if (i == nProcs - 1) {
					max = smiles.length;
				}
				try {
					FileWriter fw = new FileWriter("tmpForRDKit_" + i + ".txt");
					for (int j = min; j < max; j++) {
						fw.write(smiles[j] + "\n");
					}
					fw.close();
				} catch (IOException e) {
					throw (new RuntimeException(e.getMessage()));
				}
			}

			int[] ints = new int[nProcs];
			for (int i = 0; i < nProcs; i++) {
				ints[i] = i;
			}
			if (SystemUtils.IS_OS_LINUX || SystemUtils.IS_OS_MAC_OSX) {
				Arrays.stream(ints).parallel().forEach(i -> {
					ProcessBuilder p = new ProcessBuilder("./python/bin/python3", pythonScriptFileToRunRDKit(), "" + i)
							.inheritIO();
					try {
						Process pr = p.start();
						pr.waitFor();
					} catch (Exception e) {
						throw (new RuntimeException(e.getMessage()));
					}
				});
			}
			for (int i = 0; i < nProcs; i++) {
				try {
					BufferedReader smiFile = new BufferedReader(new FileReader("tmpForRDKit_" + i + ".txt"));
					BufferedReader outFile = new BufferedReader(new FileReader("tmpForRDKit_out_" + i + ".txt"));
					String s = outFile.readLine();
					checkThatHeaderLineContainsCorrectDescriptors(s);
					s = outFile.readLine();
					String s2 = smiFile.readLine();
					ArrayList<String> smilesList = new ArrayList<String>();
					while (s2 != null) {
						if (!s2.trim().equals("")) {
							smilesList.add(s2.trim());
						}
						s2 = smiFile.readLine();
					}
					ArrayList<String[]> descriptorsList = new ArrayList<String[]>();
					while (s != null) {
						if (!s.trim().equals("")) {
							descriptorsList.add(s.trim().split("\\,"));
						}
						s = outFile.readLine();
					}
					smiFile.close();
					outFile.close();
					if ((descriptorsList.size() != smilesList.size())) {
						throw (new RuntimeException("Output of RDKit contains wrong number of non-empty lines!"));
					}
					for (int k = 0; k < smilesList.size(); k++) {
						float[] d = stringsToFloats(descriptorsList.get(k));
						if (d.length != this.getNumFeatures()) {
							throw (new RuntimeException("Output of RDKit contains wrong number of descriptors"));
						}
						this.putPrecomputed(smilesList.get(k), d);
					}
					deleteFileIfExist("tmpForRDKit_" + i + ".txt");
					deleteFileIfExist("tmpForRDKit_out_" + i + ".txt");
				} catch (Exception e) {
					throw (new RuntimeException(e.getMessage()));
				}
			}
		}
	}

	private float[] stringsToFloats(String[] s) {
		float[] result = new float[s.length];
		for (int i = 0; i < result.length; i++) {
			try {
				result[i] = Float.parseFloat(s[i]);
				if (Math.abs(Math.abs(result[i]) - 777) < 0.00001) {
					result[i] = Float.NaN;
				}
				if (Math.abs(Math.abs(result[i]) - 777 + 111) < 0.00001) {
					result[i] = Float.NaN;
				}
			} catch (Throwable e) {
				e.printStackTrace();
				result[i] = Float.NaN;
			}
		}
		return result;
	}

	private void deleteFileIfExist(String filename) {
		File f = new File(filename);
		if (f.exists()) {
			f.delete();
		}
	}

	private void checkThatHeaderLineContainsCorrectDescriptors(String s) {
		boolean t = true;
		String[] str = s.split("\\,");
		if (str.length != descriptorsNamesWithoutPrefix().length) {
			t = false;
		}
		for (int i = 0; i < str.length; i++) {
			if (!str[i].equals(descriptorsNamesWithoutPrefix()[i])) {
				t = false;
			}
		}
		if (!t) {
			throw (new RuntimeException("Output of RDKit contains wrong set of descriptors"));
		}
	}

	/**
	 * 
	 * @return feature names without prefixes (such as RDKit_). The names should
	 *         match with table headers/descriptor names in the python scripts
	 */
	public abstract String[] descriptorsNamesWithoutPrefix();

	/**
	 * Python script (specified in subclasses) should get SMILES strings from a file
	 * that is named tmpForRDKit_N.txt, where N - is number that is recieved by the
	 * python script as the first command-line argument for the python script. The
	 * file should contain SMILES, one per line, no spaces or empty lines. The
	 * python script should read the SMILES and compute descriptors. Output should
	 * be saved as comma-separated table to tmpForRDKit_out_N.txt file. For example
	 * the python script with command-line parameter 3 reads SMILES strings from
	 * tmpForRDKit_3.txt and writes descriptors in tmpForRDKit_out_3.txt. The first
	 * line of the out file is the header (it should contain comma-separated
	 * descriptor names without prefixes). The out file should contain the header
	 * and comma-separated descriptors (one molecule per line, no lines should be
	 * missed even if a problem was encountered!). Header should match the
	 * descriptorsNamesWithoutPrefix() method. Note, that special values such as 777
	 * and -777 (and so on) that are generated when RDKit fails to generate
	 * descriptors are supported. Such values are replaced with NaNs values.
	 * 
	 * @return name of python script that is used for computing the descriptors.
	 */
	public abstract String pythonScriptFileToRunRDKit();

}
