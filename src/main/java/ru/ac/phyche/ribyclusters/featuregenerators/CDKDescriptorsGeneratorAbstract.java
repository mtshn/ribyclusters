package ru.ac.phyche.ribyclusters.featuregenerators;

import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import org.openscience.cdk.exception.CDKException;

/**
 * An abstract mutli-threaded feature generator. Can be used for any CDK-based
 * features.
 *
 */
public abstract class CDKDescriptorsGeneratorAbstract extends FeaturesGenerator {

	private class DescriptorsOfOneCompound {
		public String smiles;
		public float[] d;

		public void compute() throws CDKException {
			d = descriptorsBySMILES(smiles);
		}
	}

	@Override
	public void precompute(HashSet<String> smilesStrings) {

		HashSet<DescriptorsOfOneCompound> descriptors = new HashSet<DescriptorsOfOneCompound>();
		for (String smiles : smilesStrings) {
			if (!this.precomputedForMol(smiles)) {
				DescriptorsOfOneCompound d = new DescriptorsOfOneCompound();
				d.smiles = smiles;
				descriptors.add(d);
			}
		}
		try {
			AtomicInteger i = new AtomicInteger(0);
			descriptors.parallelStream().forEach(d -> {
				try {
					i.incrementAndGet();
					if (i.get() % 1000 == 0) {
						System.out.println("Computing CDK descriptors or fingerprints... " + i);
					}
					d.compute();
				} catch (CDKException e) {
					throw new RuntimeException(e.getMessage());
				}
			});
		} catch (Throwable e) {
			e.printStackTrace();
			throw (new RuntimeException(e.getMessage()));
		}
		for (DescriptorsOfOneCompound d : descriptors) {
			putPrecomputed(d.smiles, d.d);
		}
		if (descriptors.size() != 0) {
			System.out.println("CDK descriptors were computed.");
		}
	}

	/**
	 * This method is used by subclasses.
	 * 
	 * @param smiles SMILES string (molecule)
	 * @return features for this molecule
	 * @throws CDKException if something fails
	 */
	public abstract float[] descriptorsBySMILES(String smiles) throws CDKException;
}
