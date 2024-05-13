package ru.ac.phyche.ribyclusters.featuregenerators;

import org.openscience.cdk.exception.CDKException;

import ru.ac.phyche.ribyclusters.ChemUtils;

/**
 * Feature generator that computes functional groups counters using CDK. See doc
 * for the FeaturesGenerator class for more information about usage of this
 * class. See doc for the ChemUtils class for more information about functional
 * groups. These class is a wrapper for the ChemUtils.funcGroups method.
 *
 */
public class FuncGroupsCDKGenerator extends CDKDescriptorsGeneratorAbstract {

	@Override
	public float[] descriptorsBySMILES(String smiles) throws CDKException {
		return ChemUtils.funcGroups(smiles);
	}

	@Override
	public String getName(int i) {
		return "FuncGroupsCDK_" + i;
	}

	@Override
	public int getNumFeatures() {
		return 84;
	}

}
