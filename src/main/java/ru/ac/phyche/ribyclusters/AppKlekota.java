package ru.ac.phyche.ribyclusters;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import ru.ac.phyche.ribyclusters.clusters.Clustering;
import ru.ac.phyche.ribyclusters.clusters.KMeansIterative;
import ru.ac.phyche.ribyclusters.clusters.KMeansPCA;
import ru.ac.phyche.ribyclusters.featuregenerators.PreprocessedFeaturesGenerator;
import ru.ac.phyche.ribyclusters.models.LibLinearRI;
import ru.ac.phyche.ribyclusters.models.LibSVMRI;
import ru.ac.phyche.ribyclusters.models.ModelRI;
import ru.ac.phyche.ribyclusters.models.XgbRI;

public class AppKlekota {
	public static void main(String[] args) throws Exception {
		ChemDataset metlin =ChemDataset.loadFromFile("metlin.ri");
		metlin.makeCanonicalAll(false);
		ChemDataset metlinTest = metlin.compoundsBasedSplitAndShuffle(0.1f);
		metlin.saveToFile("metlinTrain.ri");
		metlinTest.saveToFile("metlinTest.ri");
		System.exit(0);
		/*BufferedWriter fw1 = new BufferedWriter(new FileWriter("metlin.ri"));
		BufferedReader br = new BufferedReader(new FileReader("SMRT_dataset.csv"));
		
		String s = br.readLine();
		s = br.readLine();
		int i=0;
		int j=0;
		while (s!=null) {
			String[] splt=s.trim().split("\\;");
			if(splt.length==3) {
				i++;
				String inchi = splt[2].replace('"', ' ').trim();
				String smiles = ChemUtils.inchiToSmiles(inchi, false);
				String smilesCan = ChemUtils.canonical(smiles, false);
				float rt = Float.parseFloat(splt[1]);
				if (rt > 400 ) {
					j++;
					fw1.write(smilesCan+" "+rt+" 0\n");
				}
				if (i%1000==0) {
					System.out.println(i+" "+j);
				}
			}
			s = br.readLine();
		}
		System.out.println(i+" "+j);		
		fw1.close();
		br.close();
		System.exit(0);*/
		
		
		//ChemDataset nist = ChemDataset.loadFromFile("nist.ri");
		//ChemDataset nistTest = ChemDataset.loadFromFile("nist_CV_split0.ri");
		ChemDataset nistPolar = ChemDataset.loadFromFile("nistPolar_train0.ri");
		ChemDataset nistPolarTest = ChemDataset.loadFromFile("nistPolar_split0.ri");
		System.out.println(nistPolar.size());
		System.out.println(nistPolarTest.size());
		System.out.println(nistPolar.compounds().size());
		System.out.println(nistPolarTest.compounds().size());

		// ChemDataset gmd = ChemDataset.loadFromFile("gmd.ri").shuffle();
		// gmd.makeCanonicalAll(false);
		nistPolar.filterIdenticalByInchi(nistPolarTest);
		System.out.println(nistPolar.size());
		System.out.println(nistPolarTest.size());
		// nist.filterIdenticalByInchi(gmd);

		PreprocessedFeaturesGenerator g =
		FeatureGenerators.klekotaAllDescriptorsTrained(nistPolar);
		FeatureGenerators.savePreproc(g, "preproc_nist_polar.txt");
		// g = FeatureGenerators.klekotaAllDescriptors();
		//FeatureGenerators.loadPreproc(g, "preproc_klekota_nisttest_0.txt");

		ChemDataset train = nistPolar.compoundsBasedSplitAndShuffle(0.8f);
		ChemDataset val = nistPolar;
		ColumnFeatures gc = new ColumnFeatures.NoColumnFeatures();

		FileWriter fw = new FileWriter("KLEKOTA_nistPolar4");
	

		//ModelRI m1 = new LibLinearRI(gc, g, true, "./liblinear_cluster_tune_kl/0.txt", 50);
		//m1.train(train, val);
		//fw.write("LIBLINEAR " + 1 + "\n");
		//fw.write(m1.validate(nistTest) + "\n\n\n");
		//m1.save("liblinear_cluster_tune_kl/model1");
		//fw.flush();
		//int[] clNums = new int[] { 12, 25, 30, 40, 50 };
		
		ModelRI example = new LibSVMRI(gc, g, true, null, 50);
		Clustering cl = new KMeansIterative();
		cl.init(new float[] { 60, 1, 1500, 3 });
		RIByClusters v = RIByClusters.getInstance(cl, g, example, "libsvm_cluster_tune_kl_nist_polar");
		v.train(train, val);
		v.save("libsvm_cluster_tune_kl_nist_polar/model");
		fw.write("LIBSVM, max 1500 compounds, nistPolar\n");
		fw.write(v.validate(nistPolarTest) + "\n\n\n");
		fw.flush();
		
/*
		int[] maxCompounds = new int[] { 2500, 4000, 6000, 10000};

		for (int c = 0; c < maxCompounds.length; c++) {
			int maxCompound = maxCompounds[c];
			ModelRI example = new LibLinearRI(gc, g, true, null, 50);
			Clustering cl = new KMeansIterative();
			cl.init(new float[] { 60, 1, maxCompound, 3 });
			RIByClusters v = RIByClusters.getInstance(cl, g, example, "liblinear_cluster_tune_kl_metlin/" + maxCompound);
			v.train(train, val);
			v.save("liblinear_cluster_tune_kl_metlin/model" + maxCompound);
			fw.write("LIBLINEAR " + maxCompound + "\n");
			fw.write("LIBLINEAR " + maxCompound + "\n");
			fw.write("LIBLINEAR " + maxCompound + "\n");
			fw.write(v.validate(metlinTest) + "\n\n\n");
			fw.flush();
		}
*/
		//ModelRI m2 = new XgbRI(gc, g, true, "./xgboost_cluster_tune_kl/0.txt", 50);
		//m2.train(train, val);
		//fw.write("XGBOOST " + 1 + "\n");
		//fw.write(m2.validate(nistTest) + "\n\n\n");
		//m2.save("xgboost_cluster_tune_kl/model1");
		//fw.flush();

		/*for (int c = 0; c < maxCompounds.length; c++) {
			int maxCompound = maxCompounds[c];
			ModelRI example = new XgbRI(gc, g, true, null, 50);
			Clustering cl = new KMeansIterative();
			cl.init(new float[] { 60, 1, maxCompound, 3 });
			RIByClusters v = RIByClusters.getInstance(cl, g, example, "xgboost_cluster_tune_kl_it/" + maxCompound);
			v.train(train, val);
			v.save("xgboost_cluster_tune_kl_it/model" + maxCompound);
			fw.write("XGBOOST " + maxCompound + "\n");
			fw.write("XGBOOST " + maxCompound + "\n");
			fw.write("XGBOOST " + maxCompound + "\n");
			fw.write(v.validate(nistTest) + "\n\n\n");
			fw.flush();
		}*/
		fw.close();
	}
}
