package ru.ac.phyche.ribyclusters.featuregenerators;

/**
 * 
 * Feature generator that computes the MQN molecular descriptors using RDKit.
 * MQN descriptors: DOI: 10.1002/cmdc.200900317. Real implementation of the
 * descriptor calculator is made using the Python language and is contained in
 * the rdkit_mqn.py file. The rdkit_mqn.py file should be contained in the
 * working directory. See doc for the FeaturesGenerator class for more
 * information about the use of this class. See doc for the
 * RDKitDescriptorsGeneratorAbstract class for information about the python
 * location and the python script.
 */
public class MQNDescriptorsGenerator extends RDKitDescriptorsGeneratorAbstract {

	@Override
	public String[] descriptorsNamesWithoutPrefix() {
		String[] rslt = new String[getNumFeatures()];
		for (int i = 0; i < getNumFeatures(); i++) {
			rslt[i] = i + "";
		}
		return rslt;
	}

	@Override
	public String pythonScriptFileToRunRDKit() {
		return "rdkit_mqn.py";
	}

	@Override
	public String getName(int i) {
		return "MQN_" + i;
	}

	@Override
	public int getNumFeatures() {
		return 42;
	}

}
