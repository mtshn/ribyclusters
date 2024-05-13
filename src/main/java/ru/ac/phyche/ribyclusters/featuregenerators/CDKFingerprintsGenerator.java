package ru.ac.phyche.ribyclusters.featuregenerators;

import org.openscience.cdk.exception.CDKException;

import ru.ac.phyche.ribyclusters.ChemUtils;

/**
 * Feature generator that computes CDK molecular descriptors. See doc for the
 * FeaturesGenerator class for more information about usage of this class. See
 * doc for the ChemUtils class for more information about types of molecular
 * fingerprints. These class is a wrapper for the ChemUtils.fingerprints method.
 *
 */
public class CDKFingerprintsGenerator extends CDKDescriptorsGeneratorAbstract {

	private int n = 0;
	private ChemUtils.FingerprintsType fingerprintType_;

	private int len() {
		int n;
		try {
			n = ChemUtils.fingerprints("CCCC", fingerprintType_).length;
		} catch (CDKException e) {
			throw (new RuntimeException("CDK failed"));
		}
		return n;
	}

	@Override
	public float[] descriptorsBySMILES(String smiles) throws CDKException {
		return ChemUtils.fingerprints(smiles, fingerprintType_);
	}

	@Override
	public String getName(int i) {
		return fingerprintType_.toString() + "_" + i;
	}

	@Override
	public int getNumFeatures() {
		return n;
	}

	/**
	 * 
	 * @param fingerprintType fingerprint type. See ChemUtils class
	 */
	public CDKFingerprintsGenerator(ChemUtils.FingerprintsType fingerprintType) {
		fingerprintType_ = fingerprintType;
		n = len();
	}

	@SuppressWarnings("unused")
	private CDKFingerprintsGenerator() {

	}
}
