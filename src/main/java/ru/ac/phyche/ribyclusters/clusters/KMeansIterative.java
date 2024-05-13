package ru.ac.phyche.ribyclusters.clusters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.io.xml.StaxDriver;

import ru.ac.phyche.ribyclusters.ArUtls;
import smile.clustering.KMeans;
import smile.projection.PCA;

public class KMeansIterative extends Clustering {

	private PCA pca = null;
	private int nPCA = 10;
	private boolean pcaRequired = true;
	private ArrayList<KMeans> kms; // k means clustering model for each non-leaf node,null fore leafs
	private ArrayList<Integer> leafNumbers; // number of leaf of leaf node, -1 instead;
	private ArrayList<int[]> nodes; // array of next nodes for each non-leaf node, null for leafs
	private int nSplit = 3;
	private int maxEntriesInCluster = 500;

	private void testListSizes(int n) {
		if ((leafNumbers.size() != n) || (nodes.size() != n) || (kms.size() != n)) {
			throw new RuntimeException("Tree clusterization error");
		}
	}

	@Override
	public int[] train(float[][] features) {
		ArrayList<int[]> entriesAtEachNode; // array of entry numbers at each node
		boolean cont = true;
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		if (pcaRequired) {
			nPCA = Math.min(Math.min(nPCA, features.length), features[0].length);
			pca = PCA.fit(featuresDouble);
			pca.setProjection(nPCA);
			double[][] pcaD = pca.project(featuresDouble);
			if (pcaD[0].length != nPCA) {
				throw new RuntimeException("Wrong matrix dimension after PCA");
			}
			if (pcaD.length != features.length) {
				throw new RuntimeException("Wrong matrix dimension after PCA");
			}
			featuresDouble = pcaD;
		}
		nodes = new ArrayList<int[]>();
		entriesAtEachNode = new ArrayList<int[]>();
		leafNumbers = new ArrayList<Integer>();
		kms = new ArrayList<KMeans>();
		nodes.add(null);
		entriesAtEachNode.add(ArUtls.ints(features.length));
		kms.add(null);
		leafNumbers.add(-1);
		while (cont) {
			testListSizes(entriesAtEachNode.size());
			int n = kms.size();
			for (int i = 0; i < n; i++) {
				cont = false;
				if (nodes.get(i) == null) {
					if (kms.get(i) != null) {
						throw new RuntimeException("Tree clusterization error");
					}
					if (entriesAtEachNode.get(i).length > maxEntriesInCluster) {
						cont = true;
						kms.set(i, KMeans.fit(ArUtls.subarray(featuresDouble, entriesAtEachNode.get(i)), nSplit));
						int[] newNodes = new int[nSplit];
						int[] clusterNums = kms.get(i).y;
						int n1 = kms.size();
						for (int j = 0; j < nSplit; j++) {
							kms.add(null);
							leafNumbers.add(-1);
							nodes.add(null);
							ArrayList<Integer> entries = new ArrayList<Integer>();
							for (int x = 0; x < clusterNums.length; x++) {
								if (clusterNums[x] == j) {
									entries.add(entriesAtEachNode.get(i)[x]);
								}
							}
							entriesAtEachNode.add(ArUtls.toIntArray(entries));
							newNodes[j] = n1 + j;
						}
						nodes.set(i, newNodes);
					}
				}
			}
		}
		testListSizes(entriesAtEachNode.size());
		int n = kms.size();
		int cluster = 0;
		int[] result = new int[features.length];
		for (int i = 0; i < n; i++) {
			if (nodes.get(i) == null) {
				if (kms.get(i) != null) {
					throw new RuntimeException("Tree clusterization error");
				}
				leafNumbers.set(i, cluster);
				for (int j = 0; j < entriesAtEachNode.get(i).length; j++) {
					result[entriesAtEachNode.get(i)[j]] = cluster;
				}
				cluster++;
			}
		}
		this.setClustersNum(cluster);
		return result;
	}

	@Override
	public void save(String directoryName) throws IOException {
		Files.createDirectories(Paths.get(directoryName));
		FileWriter fw = new FileWriter(new File(directoryName, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directoryName, "info.txt"));
		fw.write(this.getClustersNum() + " " + (this.pcaRequired ? "1" : "0") + " " + this.nodes.size());
		fw.close();

		FileWriter fwNodes = new FileWriter(new File(directoryName, "nodes.txt"));
		FileWriter fwLeafNums = new FileWriter(new File(directoryName, "leafClusterNum.txt"));
		for (int i = 0; i < this.nodes.size(); i++) {
			if (kms.get(i) != null) {
				fw = new FileWriter(new File(directoryName, "kmeans" + i + ".xml"));
				XStream xstream = new XStream(new StaxDriver());
				xstream.toXML(kms.get(i), fw);
			}
			fw.close();
			fwLeafNums.write(leafNumbers.get(i) + " ");
			if (nodes.get(i) != null) {
				for (int j = 0; j < nodes.get(i).length; j++) {
					fwNodes.write(nodes.get(i)[j] + " ");
				}
			} else {
				fwNodes.write("null");
			}
			fwNodes.write("\n");
		}
		fwLeafNums.close();
		fwNodes.close();
		if (pcaRequired) {
			if (pca != null) {
				fw = new FileWriter(new File(directoryName, "pca.xml"));
				XStream xstream = new XStream(new StaxDriver());
				xstream.toXML(pca, fw);
				fw.close();
			}
		}
	}

	@Override
	public void load(String directoryName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(directoryName, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		if (!s.trim().equals(this.modelType())) {
			throw new RuntimeException("Wrong model type");
		}
		br = new BufferedReader(new FileReader(new File(directoryName, "info.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		String[] split = s.trim().split("\\s+");
		this.setClustersNum(Integer.parseInt(split[0]));
		this.pcaRequired = Float.parseFloat(split[1]) >= 0.5;
		int nodes = Integer.parseInt(split[2]);
		br = new BufferedReader(new FileReader(new File(directoryName, "leafClusterNum.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		split = s.trim().split("\\s+");
		if (nodes != split.length) {
			throw new RuntimeException("Wrong number of nodes");
		}
		br = new BufferedReader(new FileReader(new File(directoryName, "nodes.txt")));
		this.nodes=new ArrayList<int[]>();
		this.leafNumbers=new ArrayList<Integer>();
		this.kms=new ArrayList<KMeans>();
		for (int i = 0; i < nodes; i++) {
			s = br.readLine();
			while (s.trim().equals("")) {
				s = br.readLine();
			}
			if (!s.trim().equals("null")) {
				if (Integer.parseInt(split[i]) != -1) {
					throw new RuntimeException("Wrong number of nodes");
				}
				String[] split1 = s.trim().trim().split("\\s+");
				int[] nodes1 = new int[split1.length];
				for (int j = 0; j < nodes1.length; j++) {
					nodes1[j] = Integer.parseInt(split1[j]);
				}
				this.nodes.add(nodes1);
				XStream xstream = new XStream();
				xstream.allowTypes(new String[] {"smile.clustering.KMeans"});
				this.kms.add((KMeans) xstream.fromXML(new File(directoryName, "kmeans" + i + ".xml")));
				this.leafNumbers.add(-1);
			} else {
				if (Integer.parseInt(split[i]) == -1) {
					throw new RuntimeException("Wrong number of nodes");
				}
				this.leafNumbers.add(Integer.parseInt(split[i]));
				this.kms.add(null);
				this.nodes.add(null);
			}
		}
		if (pcaRequired) {
			XStream xstream = new XStream();
			xstream.allowTypes(new String[] {"smile.projection.PCA","smile.math.matrix.Matrix","smile.math.matrix.Matrix$1"});
			pca = (PCA) xstream.fromXML(new File(directoryName, "pca.xml"));
			this.nPCA = pca.getProjection().nrows();
		}
	}

	@Override
	public int[] predict(float[][] features) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		if (pcaRequired) {
			pca.setProjection(nPCA);
			double[][] pcaD = pca.project(featuresDouble);
			if (pcaD[0].length != nPCA) {
				throw new RuntimeException("Wrong matrix dimension after PCA");
			}
			if (pcaD.length != features.length) {
				throw new RuntimeException("Wrong matrix dimension after PCA");
			}
			featuresDouble = pcaD;
		}
		int[] result = new int[features.length];
		for (int i = 0; i < features.length; i++) {
			int curnode = 0;
			while (nodes.get(curnode) != null) {
				curnode = nodes.get(curnode)[kms.get(curnode).predict(featuresDouble[i])];
			}
			result[i] = leafNumbers.get(curnode);
		}
		return result;
	}

	@Override
	public void init(float[] parameters) {
		if (parameters.length != 4) {
			throw new RuntimeException("Wrong number of parameters. KMeansIterative has 4 parameters.");
		}
		this.nPCA = Math.round(parameters[0]);
		this.pcaRequired = parameters[1] >= 0.5 && this.nPCA > 1;
		this.maxEntriesInCluster = Math.round(parameters[2]);
		this.nSplit = Math.round(parameters[3]);
	}

	public void init() {
		init(new float[] { 10, 1, 500, 3 });
	}

	@Override
	public String modelType() {
		return "KMeansIterative";
	}

	@Override
	public String paramsNames() {
		return "nPCA pcaRequired maxEntriesInCluster nSplit";
	}
}