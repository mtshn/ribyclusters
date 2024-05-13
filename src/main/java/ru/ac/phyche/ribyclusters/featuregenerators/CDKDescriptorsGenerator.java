package ru.ac.phyche.ribyclusters.featuregenerators;

import org.openscience.cdk.exception.CDKException;

import ru.ac.phyche.ribyclusters.ChemUtils;

/**
 * Feature generator that computes CDK molecular descriptors. All 2D molecular
 * descriptors supported by CDK but two 2D descriptors: nAtomLAC and MolIP are
 * computed. 3D descriptors (descriptors which require 3D coordinates) are not
 * computed. See doc for the FeaturesGenerator class for more information.
 *
 */
public class CDKDescriptorsGenerator extends CDKDescriptorsGeneratorAbstract {
	/**
	 * This array stores names of all molecular descriptors supported by CDK but 3D
	 * descriptors (descriptors which require 3D coordinates) and two 2D
	 * descriptors: nAtomLAC and MolIP. This two descriptors are calculated
	 * unreasonable slow for many molecules.
	 */
	public static final String[] descriptors2DBut_nAtomLAC_And_MolIP = new String[] { "WPATH", "WPOL", "MLogP",
			"nSmallRings", "nAromRings", "nRingBlocks", "nAromBlocks", "nRings3", "nRings4", "nRings5", "nRings6",
			"nRings7", "nRings8", "nRings9", "SC-3", "SC-4", "SC-5", "SC-6", "VC-3", "VC-4", "VC-5", "VC-6", "HybRatio",
			"nAtom", "nAromBond", "MDEC-11", "MDEC-12", "MDEC-13", "MDEC-14", "MDEC-22", "MDEC-23", "MDEC-24",
			"MDEC-33", "MDEC-34", "MDEC-44", "MDEO-11", "MDEO-12", "MDEO-22", "MDEN-11", "MDEN-12", "MDEN-13",
			"MDEN-22", "MDEN-23", "MDEN-33", "ALogP", "ALogp2", "AMR", "khs.sLi", "khs.ssBe", "khs.ssssBe", "khs.ssBH",
			"khs.sssB", "khs.ssssB", "khs.sCH3", "khs.dCH2", "khs.ssCH2", "khs.tCH", "khs.dsCH", "khs.aaCH",
			"khs.sssCH", "khs.ddC", "khs.tsC", "khs.dssC", "khs.aasC", "khs.aaaC", "khs.ssssC", "khs.sNH3", "khs.sNH2",
			"khs.ssNH2", "khs.dNH", "khs.ssNH", "khs.aaNH", "khs.tN", "khs.sssNH", "khs.dsN", "khs.aaN", "khs.sssN",
			"khs.ddsN", "khs.aasN", "khs.ssssN", "khs.sOH", "khs.dO", "khs.ssO", "khs.aaO", "khs.sF", "khs.sSiH3",
			"khs.ssSiH2", "khs.sssSiH", "khs.ssssSi", "khs.sPH2", "khs.ssPH", "khs.sssP", "khs.dsssP", "khs.sssssP",
			"khs.sSH", "khs.dS", "khs.ssS", "khs.aaS", "khs.dssS", "khs.ddssS", "khs.sCl", "khs.sGeH3", "khs.ssGeH2",
			"khs.sssGeH", "khs.ssssGe", "khs.sAsH2", "khs.ssAsH", "khs.sssAs", "khs.sssdAs", "khs.sssssAs", "khs.sSeH",
			"khs.dSe", "khs.ssSe", "khs.aaSe", "khs.dssSe", "khs.ddssSe", "khs.sBr", "khs.sSnH3", "khs.ssSnH2",
			"khs.sssSnH", "khs.ssssSn", "khs.sI", "khs.sPbH3", "khs.ssPbH2", "khs.sssPbH", "khs.ssssPb", "nHBDon",
			"BCUTw-1l", "BCUTw-1h", "BCUTc-1l", "BCUTc-1h", "BCUTp-1l", "BCUTp-1h", "nBase", "Fsp3", "PetitjeanNumber",
			"topoShape", "nSpiroAtoms", "bpol", "VABC", "ECCEN", "VAdjMat", "JPLogP", "nAtomLC", "nAcid", "XLogP",
			"apol", "LipinskiFailures", "Zagreb", "WTPT-1", "WTPT-2", "WTPT-3", "WTPT-4", "WTPT-5", "MW", "SP-0",
			"SP-1", "SP-2", "SP-3", "SP-4", "SP-5", "SP-6", "SP-7", "VP-0", "VP-1", "VP-2", "VP-3", "VP-4", "VP-5",
			"VP-6", "VP-7", "nRotB", "FMF", "TopoPSA", "ATSc1", "ATSc2", "ATSc3", "ATSc4", "ATSc5", "SCH-3", "SCH-4",
			"SCH-5", "SCH-6", "SCH-7", "VCH-3", "VCH-4", "VCH-5", "VCH-6", "VCH-7", "C1SP1", "C2SP1", "C1SP2", "C2SP2",
			"C3SP2", "C1SP3", "C2SP3", "C3SP3", "C4SP3", "SPC-4", "SPC-5", "SPC-6", "VPC-4", "VPC-5", "VPC-6",
			"tpsaEfficiency", "nAtomP", "ATSm1", "ATSm2", "ATSm3", "ATSm4", "ATSm5", "fragC", "Kier1", "Kier2", "Kier3",
			"naAromAtom", "ATSp1", "ATSp2", "ATSp3", "ATSp4", "ATSp5", "nHBAcc", "nB", "nA", "nR", "nN", "nD", "nC",
			"nF", "nQ", "nE", "nG", "nH", "nI", "nP", "nL", "nK", "nM", "nS", "nT", "nY", "nV", "nW" };

	@Override
	public String getName(int i) {
		return "CDK_" + descriptors2DBut_nAtomLAC_And_MolIP[i];
	}

	@Override
	public int getNumFeatures() {
		return descriptors2DBut_nAtomLAC_And_MolIP.length;
	}

	@Override
	public float[] descriptorsBySMILES(String smiles) throws CDKException {
		return ChemUtils.descriptors(smiles, descriptors2DBut_nAtomLAC_And_MolIP);
	}
}
