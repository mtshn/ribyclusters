package ru.ac.phyche.ribyclusters.featuregenerators;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.DatasetEntry;

/**
 * 
 * Subclasses this class are used to generate features for a molecule for
 * further using with ML methods. It allows to create features for single SMILES
 * string, for array of them. It stores (caches) precomputed values of features
 * for each SMILES string. Note that no canonicalization are performed. SMILES
 * strings are used as is.
 *
 */
public abstract class FeaturesGenerator {
	private ConcurrentHashMap<String, float[]> precomputed = new ConcurrentHashMap<String, float[]>();

	/**
	 * Returns precomputed values. SMILES string is used as a key, float array is
	 * used as a value
	 * 
	 * @return
	 */
	public ConcurrentHashMap<String, float[]> getPrecomputedMap() {
		return precomputed;
	}

	/**
	 * 
	 * @param smiles SMILES string for a molecule
	 * @return true if precompured features are stored for this SMILES string
	 */
	public boolean precomputedForMol(String smiles) {
		return (precomputed.get(smiles.trim()) != null);
	}

	/**
	 * If there are no precomputed features for this SMILES, a RuntimeException will
	 * be thrown. Precomputed features should be cached. Use the precompute method
	 * preliminary or use featuresForMolNoPrecompute instead this method.
	 * 
	 * @param smiles SMILES string for a molecule
	 * @return features for this SMILES string
	 */
	public float[] featuresForMol(String smiles) {
		float[] result = precomputed.get(smiles.trim());
		if (result == null) {
			throw (new RuntimeException("Features MUST be precomputed before call the featuresForMol method;"
					+ "use the precompute method or featuresForMolNoPrecompute instead"));
		} else {
			return result;
		}
	}

	/**
	 * If there are no precomputed features for this SMILES, a RuntimeException will
	 * be thrown. Precomputed features should be cached. Use the precompute method
	 * preliminary or use featuresForMolNoPrecompute instead this method.
	 * 
	 * @param mol molecule (only SMILES string from this data structure is actually
	 *            used)
	 * @return features for this molecule (SMILES string)
	 */
	public float[] featuresForMol(DatasetEntry mol) {
		return featuresForMol(mol.getSmiles());
	}

	/**
	 * The computed value will be cached. Use precomputing and featuresForMol for
	 * the best performance instead this method.
	 * 
	 * @param smiles SMILES string for a molecule
	 * @return features for this SMILES string
	 */
	public float[] featuresForMolNoPrecompute(String smiles) {
		precompute(new String[] { smiles });
		return featuresForMol(smiles);
	}

	/**
	 * The computed value will be cached. Use precomputing and featuresForMol for
	 * the best performance instead this method.
	 * 
	 * @param mol molecule (only SMILES string from this data structure is actually
	 *            used)
	 * @return features for this molecule (SMILES string)
	 */
	public float[] featuresForMolNoPrecompute(DatasetEntry mol) {
		precompute(new String[] { mol.getSmiles() });
		return featuresForMol(mol);
	}

	/**
	 * If there are no precomputed features for this SMILES, a RuntimeException will
	 * be thrown. Precomputed features should be cached. Use the precompute method
	 * preliminary or use featuresNoPrecompute instead this method.
	 * 
	 * @param smiles SMILES strings for molecules
	 * @return features for these SMILES strings
	 */
	public float[][] features(String[] smiles) {
		float[][] result = new float[smiles.length][];
		for (int i = 0; i < smiles.length; i++) {
			float[] result_i = precomputed.get(smiles[i]);
			if (result_i == null) {
				throw (new RuntimeException("Features MUST be precomputed before call the features method;"
						+ "use the precompute method or featuresNoPrecompute instead"));
			} else {
				result[i] = result_i;
			}
		}
		return result;
	}

	/**
	 * If there are no precomputed features for this SMILES, a RuntimeException will
	 * be thrown. Precomputed features should be cached. Use the precompute method
	 * preliminary or use featuresNoPrecompute instead this method.
	 * 
	 * @param mols molecules (only SMILES strings from these data are actually used)
	 * @return features for these molecules
	 */
	public float[][] features(DatasetEntry[] mols) {
		float[][] result = new float[mols.length][];
		for (int i = 0; i < mols.length; i++) {
			float[] result_i = precomputed.get(mols[i].getSmiles());
			if (result_i == null) {
				System.out.println(mols[i].getSmiles());
				throw (new RuntimeException("Features MUST be precomputed before call the features method;"
						+ "use the precompute method or featuresNoPrecompute instead"));
			} else {
				result[i] = result_i;
			}
		}
		return result;
	}

	/**
	 * If there are no precomputed features for this SMILES, a RuntimeException will
	 * be thrown. Precomputed features should be cached. Use the precompute method
	 * preliminary or use featuresNoPrecompute instead this method.
	 * 
	 * @param dataset molecules (only SMILES strings from these data are actually
	 *                used)
	 * @return features for these molecules
	 */
	public float[][] features(ChemDataset dataset) {
		DatasetEntry[] mols = dataset.getData();
		return features(mols);
	}

	/**
	 * The computed values will be cached. Use precomputing and the features method
	 * for the best performance instead this method.
	 * 
	 * @param smiles SMILES strings for molecules
	 * @return features for these SMILES strings
	 */
	public float[][] featuresNoPrecompute(String[] smiles) {
		HashSet<String> smilesStringsForPrecomputaion = new HashSet<String>();
		for (int i = 0; i < smiles.length; i++) {
			float[] result_i = precomputed.get(smiles[i]);
			if (result_i == null) {
				smilesStringsForPrecomputaion.add(smiles[i]);
			}
		}
		precompute(smilesStringsForPrecomputaion);
		return features(smiles);
	}

	/**
	 * The computed values will be cached. Use precomputing and the features method
	 * for the best performance instead this method.
	 * 
	 * @param mols molecules (SMILES string will be used)
	 * @return features for these molecules
	 */
	public float[][] featuresNoPrecompute(DatasetEntry[] mols) {
		HashSet<String> smilesStringsForPrecomputaion = new HashSet<String>();
		for (int i = 0; i < mols.length; i++) {
			float[] result_i = precomputed.get(mols[i].getSmiles());
			if (result_i == null) {
				smilesStringsForPrecomputaion.add(mols[i].getSmiles());
			}
		}
		precompute(smilesStringsForPrecomputaion);
		return features(mols);
	}

	/**
	 * The computed values will be cached. Use precomputing and the features method
	 * for the best performance instead this method.
	 *
	 * @param dataset molecules (SMILES string will be used)
	 * @return features for these molecules
	 */
	public float[][] featuresNoPrecompute(ChemDataset dataset) {
		DatasetEntry[] mols = dataset.getData();
		return featuresNoPrecompute(mols);
	}

	/**
	 * Save precomputed (cached) values as a file. File format: one line per SMILES.
	 * Space separated values. SMILES string, space, feature 1, space, feature 2,
	 * space etc. No header, no empty lines, no comments.
	 * 
	 * @param fileName file name
	 * @throws IOException io exception
	 */
	public void savePrecomputed(String fileName) throws IOException {
		FileWriter fw = new FileWriter(fileName);
		for (String s : precomputed.keySet()) {
			fw.write(s + " ");
			float[] d = precomputed.get(s);
			for (int i = 0; i < d.length; i++) {
				fw.write(d[i] + " ");
			}
			fw.write("\n");
		}
		fw.close();
	}

	/**
	 * Load precomputed (cached) values as a file. File format: one line per SMILES.
	 * Space separated values. SMILES string, space, feature 1, space, feature 2,
	 * space etc. No header, no empty lines, no comments.
	 * 
	 * @param fileName file name
	 * @throws IOException io exception
	 */
	public void loadPrecomputed(String fileName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String s = br.readLine();
		while (s != null) {
			s = s.trim();
			if (!s.equals("")) {
				String[] split = s.split("\\s+");
				String smiles = split[0];
				float[] features = new float[split.length - 1];
				for (int i = 0; i < features.length; i++) {
					features[i] = Float.parseFloat(split[i + 1].trim());
				}
				precomputed.put(smiles, features);
			}
			s = br.readLine();
		}
		br.close();
	}

	/**
	 * Precompute features for each SMILES from the data set.
	 * 
	 * @param data data set
	 */
	public void precompute(ChemDataset data) {
		HashSet<String> smilesSet = new HashSet<String>();
		for (int i = 0; i < data.size(); i++) {
			smilesSet.add(data.getSmiles(i));
		}
		precompute(smilesSet);
	}

	/**
	 * Precompute features for each SMILES from the array.
	 * 
	 * @param smilesStrings SMILES strings
	 */
	public void precompute(String[] smilesStrings) {
		HashSet<String> smilesSet = new HashSet<String>();
		for (int i = 0; i < smilesStrings.length; i++) {
			smilesSet.add(smilesStrings[i]);
		}
		precompute(smilesSet);
	}

	/**
	 * Cache given feature values as precomputed. This is used by subclasses in
	 * implementations of the precompute method.
	 * 
	 * @param smiles   SMILES string (key)
	 * @param features features.
	 */
	public void putPrecomputed(String smiles, float[] features) {
		precomputed.put(smiles, features);
	}

	/**
	 * This method provides access to names of features. One string per one feature.
	 * String name correspond to each float feature.
	 * 
	 * @return names of features.
	 */
	public String[] getNames() {
		int n = getNumFeatures();
		String[] result = new String[n];
		for (int i = 0; i < result.length; i++) {
			result[i] = getName(i);
		}
		return result;
	}

	/**
	 * 
	 * @return number of SMILES for those features are precomputed.
	 */
	public int precomputedSize() {
		return precomputed.size();
	}

	/**
	 * Precompute features for each SMILES from the set.
	 * 
	 * @param smilesStrings SMILES strings
	 */
	public abstract void precompute(HashSet<String> smilesStrings);

	/**
	 * This method provides access to names of features. String name correspond to
	 * each float feature.
	 * 
	 * @param i number of feature
	 * @return name of feature
	 */
	public abstract String getName(int i);

	/**
	 * 
	 * @return number of features (fixed number for any SMILES string)
	 */
	public abstract int getNumFeatures();

}