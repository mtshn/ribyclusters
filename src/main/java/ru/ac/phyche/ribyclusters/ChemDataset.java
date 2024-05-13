package ru.ac.phyche.ribyclusters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Map.Entry;

import org.openscience.cdk.exception.CDKException;

public class ChemDataset {
	private DatasetEntry[] data;

	/**
	 * 
	 * @return Number of entries (value - SMILES) in the data set.
	 */
	public int size() {
		return data.length;
	}

	/**
	 * Returns complete data set as array of DatasetEntry
	 * 
	 * @return DatasetEntry[]
	 */
	public DatasetEntry[] getData() {
		return data;
	}

	/**
	 * 
	 * @param data complete data set as array of DatasetEntry
	 */
	public void setData(DatasetEntry[] data) {
		this.data = data;
	}

	/**
	 * getData()[i]
	 * 
	 * @param i number of entry to return
	 * @return i-th entry
	 */
	public DatasetEntry getEntry(int i) {
		return data[i];
	}

	/**
	 * 
	 * @param i number of entry
	 * @return SMILES of i-th entry
	 */
	public String getSmiles(int i) {
		return data[i].getSmiles();
	}

	/**
	 * 
	 * @param i number of entry
	 * @return value of property for i-th entry
	 */
	public float getRetention(int i) {
		return data[i].getRetention();
	}

	public int getColumn(int i) {
		return data[i].getColumn();
	}

	/**
	 * data[i] = e;
	 * 
	 * @param e entry
	 * @param i number of entry
	 */
	public void setEntry(DatasetEntry e, int i) {
		data[i] = e;
	}

	/**
	 * Create new instance from array
	 * 
	 * @param data_ array of DatasetEntry
	 * @return new instance
	 */
	public static ChemDataset create(DatasetEntry[] data_) {
		ChemDataset result = new ChemDataset();
		result.data = data_;
		return result;
	}

	/**
	 * Create new instance from ArrayList
	 * 
	 * @param data_ list of entries
	 * @return new instance
	 */
	public static ChemDataset create(ArrayList<DatasetEntry> data_) {
		ChemDataset result = new ChemDataset();
		result.data = data_.toArray(new DatasetEntry[data_.size()]);
		return result;
	}

	/**
	 * Deep copy of the data set (with copying in memory of all records)
	 * 
	 * @return deep copy
	 */
	public ChemDataset copy() {
		ChemDataset result = new ChemDataset();
		result.data = new DatasetEntry[this.data.length];
		for (int i = 0; i < this.data.length; i++) {
			result.data[i] = this.data[i].deepclone();
		}
		return result;
	}

	/**
	 * Shuffle this data set. Change order of entries.
	 * 
	 * @return this
	 */
	public ChemDataset shuffle() {
		Random rnd = new Random();
		for (int c = 0; c < 10; c++) {
			for (int i = 0; i < this.data.length; i++) {
				int j = rnd.nextInt(this.data.length);
				DatasetEntry a = data[j];
				data[j] = data[i];
				data[i] = a;
			}
		}
		return this;
	}

	/**
	 * This method allows to obtain a set of compounds which are conatined in this
	 * data set. Each compounds is contained only once (SMILES are stored in the set
	 * in "canonical" form). If the data set contains various SMILES strings for
	 * really one compound, it will be contained in the HashSet only once.
	 * 
	 * @param stereochemistry treat cis/trans and optical compounds as different
	 *                        (TRUE) or as identical (FALSE).
	 * @return HashSet of SMILES strings
	 * @throws CDKException CDK error during canonicalization
	 */
	public HashSet<String> compoundsCanonical(boolean stereochemistry) throws CDKException {
		HashSet<String> result = new HashSet<String>();
		for (int i = 0; i < this.data.length; i++) {
			result.add(ChemUtils.canonical(data[i].getSmiles(), stereochemistry).trim());
		}
		return result;
	}

	/**
	 * 
	 * @return HashSet of SMILES strings from this data set. No canonicalization is
	 *         used. It can contain actually more than one record for a structure
	 *         due to ambiguity of SMILES representation.
	 */
	public HashSet<String> compounds() {
		HashSet<String> result = new HashSet<String>();
		for (int i = 0; i < this.data.length; i++) {
			result.add(data[i].getSmiles().trim());
		}
		return result;
	}

	/**
	 * 
	 * @return HashSet of InChI strings for compounds from this data set.
	 *         Stereoisomers will be treated as DIFFERENT(!!!) compound if
	 *         stereoisomers are already stored with symbols denoting
	 *         stereochemistry.
	 * @throws CDKException CDK error
	 */
	public HashSet<String> inchiIds() throws CDKException {
		HashSet<String> result = new HashSet<String>();
		for (int i = 0; i < this.data.length; i++) {
			result.add(ChemUtils.smilesToInchi(data[i].getSmiles()).trim());
		}
		return result;
	}

	/**
	 * 
	 * @return HashSet of InChI-key strings for compounds from this data set.
	 *         Stereoisomers will be treated as DIFFERENT(!!!) compound if
	 *         stereoisomers are already stored with symbols denoting
	 *         stereochemistry.
	 * @throws CDKException CDK error
	 */
	public HashSet<String> inchiKeys() throws CDKException {
		HashSet<String> result = new HashSet<String>();
		for (int i = 0; i < this.data.length; i++) {
			result.add(ChemUtils.smilesToInchiKey(data[i].getSmiles()).trim());
		}
		return result;
	}

	/**
	 * Comparison using InChI strings
	 * 
	 * @param second other data set
	 * @return number of compounds which are contained in both data sets
	 *         simultaneously. Stereoisomers will be treated as DIFFERENT(!!!)
	 *         compound if stereoisomers are already stored with symbols denoting
	 *         stereochemistry. CDK
	 * @throws CDKException CDK
	 */
	public int countIdenticalByInchi(ChemDataset second) throws CDKException {
		HashSet<String> a = this.inchiIds();
		HashSet<String> b = second.inchiIds();
		a.retainAll(b);
		return a.size();
	}

	/**
	 * Comparison using InChI-key strings
	 * 
	 * @param second other data set
	 * @return number of compounds which are contained in both data sets
	 *         simultaneously. Stereoisomers will be treated as DIFFERENT(!!!)
	 *         compound if stereoisomers are already stored with symbols denoting
	 *         stereochemistry.
	 * @throws CDKException CDK
	 */
	public int countIdenticalByInchikeys(ChemDataset second) throws CDKException {
		HashSet<String> a = this.inchiKeys();
		HashSet<String> b = second.inchiKeys();
		a.retainAll(b);
		return a.size();
	}

	/**
	 * Comparison using "canonical" SMILES strings
	 * 
	 * @param second other data set
	 * @return number of compounds which are contained in both data sets
	 *         simultaneously. Stereoisomers will be treated as DIFFERENT(!!!)
	 *         compound if stereoisomers are already stored with symbols denoting
	 *         stereochemistry.
	 * @throws CDKException CDK error
	 */
	public int countIdenticalByCanonicalSmiles(ChemDataset second) throws CDKException {
		HashSet<String> a = this.compoundsCanonical(true);
		HashSet<String> b = second.compoundsCanonical(true);
		a.retainAll(b);
		return a.size();
	}

	/**
	 * This methods calls countIdenticalByInchi method for each data set from array
	 * d1 and each data set from array d2.
	 * 
	 * @param d1 array of data sets
	 * @param d2 array of data sets
	 * @return int[d1.length][d2.length];
	 * @throws CDKException CDK error
	 */
	public static int[][] countIdenticalByInchi(ChemDataset[] d1, ChemDataset[] d2) throws CDKException {
		int[][] result = new int[d1.length][d2.length];
		for (int i = 0; i < d1.length; i++) {
			for (int j = 0; j < d2.length; j++) {
				result[i][j] = d1[i].countIdenticalByInchi(d2[j]);
			}
		}
		return result;
	}

	private HashSet<String> splitSet(HashSet<String> fullSet, int n) {
		String[] a = fullSet.toArray(new String[fullSet.size()]);
		Random rnd = new Random();
		for (int c = 1; c < 10; c++) {
			for (int i = 0; i < a.length; i++) {
				int j = rnd.nextInt(a.length);
				String b = a[j];
				a[j] = a[i];
				a[i] = b;
			}
		}
		HashSet<String> result = new HashSet<String>();
		for (int i = 0; i < n; i++) {
			result.add(a[i]);
		}
		return result;
	}

	/**
	 * Shuffle and split data sets such way that all entries corresponding to any
	 * compound will be contained in only one of subsets. I.e. will be no compounds
	 * which are contained in both subsets simultaneously. Note, that no
	 * canonicalization of SMILES is used here. If in data set there are cis/trans
	 * or optical isomers they will be treated as different compounds. Use
	 * makeCanonical(false) to remove such occurrences.
	 * 
	 * @param compoundsToSplit number of COMPOUNDS (!!!not data entries!!!) which
	 *                         will be separated to the subset.
	 * @return subset which contains all data entries for compoundsToSplit compounds
	 * @throws CDKException CDK error
	 */
	public ChemDataset compoundsBasedSplitAndShuffle(int compoundsToSplit) {
		this.shuffle();
		HashSet<String> split = splitSet(this.compounds(), compoundsToSplit);
		ArrayList<DatasetEntry> splitData = new ArrayList<DatasetEntry>();
		ArrayList<DatasetEntry> retainData = new ArrayList<DatasetEntry>();
		for (int i = 0; i < this.data.length; i++) {
			if (split.contains(data[i].getSmiles().trim())) {
				splitData.add(this.getEntry(i));
			} else {
				retainData.add(this.getEntry(i));
			}
		}
		ChemDataset result = new ChemDataset();
		result.data = splitData.toArray(new DatasetEntry[splitData.size()]);
		this.data = retainData.toArray(new DatasetEntry[retainData.size()]);
		result.shuffle();
		this.shuffle();
		return result;
	}

	/**
	 * The same as compoundsBasedSplitAndShuffle(int compoundsToSplit) but fraction
	 * of compounds that should be separated is given instead number of compounds.
	 * I.e if all data set contains 1000 compounds and fraction = 0.25 it means that
	 * 250 compounds will be isolated and 750 will be remained.
	 * 
	 * @param fraction number of compounds which will be in split in all compounds
	 * @return subset which contains all data entries for Math.round(fraction *
	 *         ((float) this.compounds().size())) compounds
	 * @throws CDKException CDK error
	 */
	public ChemDataset compoundsBasedSplitAndShuffle(float fraction) throws CDKException {
		int n = this.compounds().size();
		int split = Math.round(fraction * ((float) n));
		ChemDataset result = this.compoundsBasedSplitAndShuffle(split);
		return result;
	}

	/**
	 * Simple split (based on ENTRIES (records), not compounds. Many records can
	 * correspond to one compound. This method doesn't shuffle the data set. FIRST
	 * toSplit entries will be separated. this.size() - toSplit will be remained
	 * 
	 * @param toSplit how many entries should be separated.
	 * @return data set consisted of first toSplit records. This records will be
	 *         removed from this data set.
	 */
	public ChemDataset simpleSplit(int toSplit) {
		ArrayList<DatasetEntry> splitData = new ArrayList<DatasetEntry>();
		ArrayList<DatasetEntry> retainData = new ArrayList<DatasetEntry>();
		for (int i = 0; i < this.data.length; i++) {
			if (i < toSplit) {
				splitData.add(this.getEntry(i));
			} else {
				retainData.add(this.getEntry(i));
			}
		}
		ChemDataset result = new ChemDataset();
		result.data = splitData.toArray(new DatasetEntry[splitData.size()]);
		this.data = retainData.toArray(new DatasetEntry[retainData.size()]);
		return result;
	}

	/**
	 * Simple split (based on ENTRIES (records), not compounds. Many records can
	 * correspond to one compound. This method doesn't shuffle te data set. FIRST
	 * Math.round(fraction * ((float) this.size())) entries will be separated.
	 * 
	 * @param fraction fraction of the data set to be separated
	 * @return data set with first Math.round(fraction * ((float) this.size()))
	 *         records. This records will be removed from this data set.
	 */
	public ChemDataset simpleSplit(float fraction) {
		int n = this.size();
		int split = Math.round(fraction * ((float) n));
		ChemDataset result = this.simpleSplit(split);
		return result;
	}

	/**
	 * @param toSplit how many entries will be separated after shuffling
	 * @return set consisted of toSplit entries. This records will be removed from
	 *         this data set.
	 */
	public ChemDataset simpleShuffleSplit(int toSplit) {
		this.shuffle();
		ChemDataset result = this.simpleSplit(toSplit);
		return result;
	}

	/**
	 * 
	 * @param fraction fraction of the data set to be separated
	 * @return data set with first Math.round(fraction * ((float) this.size()))
	 *         records. This records will be removed from this data set.
	 */
	public ChemDataset simpleShuffleSplit(float fraction) {
		this.shuffle();
		ChemDataset result = this.simpleSplit(fraction);
		return result;
	}

	/**
	 * Remove from THIS data set all compounds that are contained in second data
	 * set! After this method will be no compounds in both data set simultaneously.
	 * If in data set there are cis/trans or optical isomers, they will be treated
	 * as different compounds. Use makeCanonical(false).
	 * 
	 * @param second another data set.
	 * @throws CDKException CDK errors
	 */
	public void filterIdentical(ChemDataset second) throws CDKException {
		ArrayList<DatasetEntry> retainData = new ArrayList<DatasetEntry>();
		HashSet<String> b = second.compoundsCanonical(true);
		for (int i = 0; i < this.data.length; i++) {
			if (!b.contains(ChemUtils.canonical(data[i].getSmiles(), true).trim())) {
				retainData.add(this.getEntry(i));
			}
		}
		this.data = retainData.toArray(new DatasetEntry[retainData.size()]);
	}

	/**
	 * Remove from THIS data set all compounds that are contained in second data
	 * set! After this method will be no compounds in both data set simultaneously.
	 * If in data set there are cis/trans or optical isomers, they will be treated
	 * as different compounds. Use makeCanonical(false).
	 * 
	 * @param second another data set.
	 * @throws CDKException CDK errors
	 */
	public void filterIdenticalByInchi(ChemDataset second) throws CDKException {
		ArrayList<DatasetEntry> retainData = new ArrayList<DatasetEntry>();
		HashSet<String> b = second.inchiIds();
		for (int i = 0; i < this.data.length; i++) {
			if (!b.contains(ChemUtils.smilesToInchi(data[i].getSmiles()).trim())) {
				retainData.add(this.getEntry(i));
			}
		}
		this.data = retainData.toArray(new DatasetEntry[retainData.size()]);
	}

	/**
	 * Save whole data set to file. File format: one line per entry, no empty lines,
	 * no comments. Each line contains SMILES string and value of property. SMILES,
	 * the property value are separated by spaces. Example of line: "CCCC 400".
	 * 
	 * @param filename name of file (or path) for saving
	 * @throws IOException IO
	 */
	public void saveToFile(String filename) throws IOException {
		FileWriter fw = new FileWriter(filename);
		for (int i = 0; i < this.data.length; i++) {
			fw.write(this.getSmiles(i) + " " + this.getRetention(i) + " " + this.getColumn(i) + "\n");
		}
		fw.close();
	}

	/**
	 * Load data set from file. File format: one line per entry, no empty lines, no
	 * comments. Each line contains SMILES string and value of property (float).
	 * SMILES and the property values are separated by spaces. Example of line:
	 * "CCCC 400". SMILES strings loaded "as is", without checking or converting to
	 * a canonical form.
	 * 
	 * @param filename file name
	 * @return loaded data set
	 * @throws IOException IO
	 */
	public static ChemDataset loadFromFile(String filename) throws IOException {
		BufferedReader inp = new BufferedReader(new InputStreamReader(new FileInputStream(new File(filename))));
		ArrayList<DatasetEntry> data = new ArrayList<DatasetEntry>();
		String s = inp.readLine();
		while ((s != null) && (!s.trim().equals(""))) {
			String[] spl = s.split("\\s+");
			data.add(DatasetEntry.instance(spl[0], Float.parseFloat(spl[1]), Integer.parseInt(spl[2])));
			s = inp.readLine();
		}
		inp.close();
		return ChemDataset.create(data);
	}

	/**
	 * Merge multiple data sets into one
	 * 
	 * @param a array of data sets
	 * @return merged data set which contains all entries from all data sets from a
	 */
	public static ChemDataset merge(ChemDataset[] a) {
		ArrayList<DatasetEntry> data = new ArrayList<DatasetEntry>();
		for (int i = 0; i < a.length; i++) {
			data.addAll(Arrays.asList(a[i].getData()));
		}
		return ChemDataset.create(data);
	}

	/**
	 * Convert all SMILES strings in this data set to canonical form. See method
	 * canonical from class Chemoinformatics for more information.
	 * 
	 * @param stereochemistry if TRUE - cis/trans isomers and optical isomers will
	 *                        be denoted using special symbols. If false - cis/trans
	 *                        and optical isomers will be identical.
	 * @throws CDKException CDK error
	 */
	public ChemDataset makeCanonicalAll(boolean stereochemistry) throws CDKException {
		for (int i = 0; i < this.data.length; i++) {
			this.getEntry(i).setSmiles(ChemUtils.canonical(this.getSmiles(i), stereochemistry));
		}
		return this;
	}

	/**
	 * The data set can contain multiple entries for each compound (for each SMILES
	 * string). This method groups all entries for each compound together. For
	 * example if we have a data set with 6 entries "CCC 300 0","CCC 301 1", "CCCC
	 * 400 0","CCC 302 2", "CCCCC 500 0", "CCCC 401 1", this method with return
	 * HashMap with three elements. Key "CCC", ArrayList with three entries ("CCC
	 * 300 0","CCC 301 1", "CCC 302 2"); Key "CCC", ArrayList with two entries
	 * ("CCCC 400 0","CCCC 401 1"); Key "CCCCC", ArrayList with one entry ("CCCCC
	 * 500 0"). SMILES are converted to a canonical form! I.e. (C)CC and CCC will be
	 * grouped together.
	 * 
	 * @param stereochemistry if TRUE - cis/trans isomers and optical isomers will
	 *                        be considered as different compounds. If false -
	 *                        cis/trans and optical isomers will be identical.
	 * @return HashMap. Key - SMILES string, element - all entries with this
	 *         compound.
	 * @throws CDKException error during creation of canonical form.
	 */
	public HashMap<String, ArrayList<DatasetEntry>> groupByCompounds(boolean stereochemistry) throws CDKException {
		HashMap<String, ArrayList<DatasetEntry>> result = new HashMap<String, ArrayList<DatasetEntry>>();
		HashSet<String> compounds = this.compoundsCanonical(stereochemistry);
		for (String smi : compounds) {
			result.put(smi, (new ArrayList<DatasetEntry>()));
		}
		for (int i = 0; i < this.data.length; i++) {
			ArrayList<DatasetEntry> c = result.get(ChemUtils.canonical(this.getSmiles(i), stereochemistry));
			c.add(this.getEntry(i));
		}
		return result;
	}

	/**
	 * The data set can contain multiple entries for each compound (for each SMILES
	 * string). This method groups all entries for each compound together. For
	 * example if we have a data set with 6 entries "CCC 300","CCC 301", "CCCC
	 * 400","CCC 302", "CCCCC 500", "CCCC 401", this method will return ChemDataset
	 * with three entries: "CCCC 301","CCCC 400.5", "CCCCC 500". All values for each
	 * compound will be averaged together. Result will have as many entries how many
	 * really different compounds has this data set.
	 * 
	 * @param stereochemistry if TRUE - cis/trans isomers and optical isomers will
	 *                        be considered as different compounds. If false -
	 *                        cis/trans and optical isomers will be identical.
	 * @return ChemDataset that has one entry per compound.
	 * @throws CDKException error during creation of canonical form.
	 */
	public ChemDataset meanByCompounds(boolean stereochemistry) throws CDKException {
		return this.meanOrMedianByCompounds(stereochemistry, false, 0);
	}

	/**
	 * The same as meanByCompounds but median value will be calculated for each
	 * compound instead average.
	 * 
	 * @param stereochemistry if TRUE - cis/trans isomers and optical isomers will
	 *                        be considered as different compounds. If false -
	 *                        cis/trans and optical isomers will be identical.
	 * @return ChemDataset that has one entry per compound.
	 * @throws CDKException error during creation of canonical form.
	 */
	public ChemDataset medianByCompounds(boolean stereochemistry) throws CDKException {
		return this.meanOrMedianByCompounds(stereochemistry, true, 0);
	}

	public ChemDataset medianByCompounds(boolean stereochemistry, int minRecords) throws CDKException {
		return this.meanOrMedianByCompounds(stereochemistry, true, minRecords);
	}

	public ChemDataset meanByCompounds(boolean stereochemistry, int minRecords) throws CDKException {
		return this.meanOrMedianByCompounds(stereochemistry, false, minRecords);
	}

	private ChemDataset meanOrMedianByCompounds(boolean stereochemistry, boolean median, int minRecords)
			throws CDKException {
		HashMap<String, ArrayList<DatasetEntry>> a = this.groupByCompounds(stereochemistry);
		ArrayList<DatasetEntry> data = new ArrayList<DatasetEntry>();
		for (Entry<String, ArrayList<DatasetEntry>> e : a.entrySet()) {
			ArrayList<DatasetEntry> entries = e.getValue();
			float[] values = new float[entries.size()];
			for (int i = 0; i < entries.size(); i++) {
				values[i] = entries.get(i).getRetention();
			}
			float meanVal = median ? ArUtls.median(values) : ArUtls.mean(values);
			if (values.length >= minRecords) {
				data.add(DatasetEntry.instance(e.getKey(), meanVal, -1));
			}
		}
		ChemDataset result = ChemDataset.create(data);
		return result;
	}

	public static String[] smilesFromChemDataset(ChemDataset data) {
		String[] result = new String[data.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = data.getEntry(i).getSmiles();
		}
		return result;
	}

	public static int[] columnsFromChemDataset(ChemDataset data) {
		int[] result = new int[data.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = data.getEntry(i).getColumn();
		}
		return result;
	}

	public static float[] retentionsFromChemDataset(ChemDataset data) {
		float[] result = new float[data.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = data.getEntry(i).getRetention();
		}
		return result;
	}

	public String[] allSmiles() {
		return smilesFromChemDataset(this);
	}

	public int[] allColumns() {
		return columnsFromChemDataset(this);
	}

	public float[] allRetentions() {
		return retentionsFromChemDataset(this);
	}

	public static ChemDataset merge(ChemDataset a, ChemDataset b) {
		ArrayList<DatasetEntry> x = new ArrayList<DatasetEntry>();
		for (int i = 0; i < a.size(); i++) {
			x.add(a.getEntry(i));
		}
		for (int i = 0; i < b.size(); i++) {
			x.add(b.getEntry(i));
		}
		return ChemDataset.create(x);
	}

	public static ChemDataset merge(ArrayList<ChemDataset> a) {
		ArrayList<DatasetEntry> x = new ArrayList<DatasetEntry>();
		for (int i = 0; i < a.size(); i++) {
			for (int j = 0; j < a.get(i).size(); j++) {
				x.add(a.get(i).getEntry(j));
			}
		}
		return ChemDataset.create(x);
	}

	public static ChemDataset empty() {
		return ChemDataset.create(new ArrayList<DatasetEntry>());
	}

}
