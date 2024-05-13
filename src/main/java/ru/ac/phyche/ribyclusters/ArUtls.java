package ru.ac.phyche.ribyclusters;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Problem;
import libsvm.svm_node;
import libsvm.svm_problem;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import smile.data.DataFrame;
import smile.data.type.StructType;

public class ArUtls {

	public static double[] toDoubleArray(float[] x) {
		double[] result = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			result[i] = x[i];
		}
		return result;
	}

	public static double[] toDoubleArray(Float[] x) {
		double[] result = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			result[i] = x[i];
		}
		return result;
	}

	public static double[][] subarray(double[][]a, int[] indicesToSeparate){
		double[][] result = new double[indicesToSeparate.length][];
		for(int i=0;i<indicesToSeparate.length;i++) {
			result[i]=a[indicesToSeparate[i]];
		}
		return result;
	}
	
	public static double[] toDoubleArray(ArrayList<Float> x) {
		double[] result = new double[x.size()];
		for (int i = 0; i < x.size(); i++) {
			result[i] = x.get(i);
		}
		return result;
	}

	public static double[][] toDoubleArray2d(float[] x) {
		double[][] result = new double[x.length][];
		for (int i = 0; i < x.length; i++) {
			result[i] = new double[1];
			result[i][0] = x[i];
		}
		return result;
	}

	public static double[][] toDoubleArray2d(float[][] x) {
		double[][] result = new double[x.length][];
		for (int i = 0; i < x.length; i++) {
			result[i] = new double[x[i].length];
			for (int j = 0; j < x[i].length; j++) {
				result[i][j] = x[i][j];
			}
		}
		return result;
	}

	public static double[][] toDoubleArray2d(Float[][] x) {
		double[][] result = new double[x.length][];
		for (int i = 0; i < x.length; i++) {
			result[i] = new double[x[i].length];
			for (int j = 0; j < x[i].length; j++) {
				result[i][j] = x[i][j];
			}
		}
		return result;
	}

	public static double[][] toDoubleArray2d(ArrayList<float[]> x) {
		double[][] result = new double[x.size()][];
		for (int i = 0; i < x.size(); i++) {
			result[i] = new double[x.get(i).length];
			for (int j = 0; j < x.get(i).length; j++) {
				result[i][j] = x.get(i)[j];
			}
		}
		return result;
	}

	public static float[][] toFloatArray2d(float[] x) {
		float[][] result = new float[x.length][];
		for (int i = 0; i < x.length; i++) {
			result[i] = new float[1];
			result[i][0] = x[i];
		}
		return result;
	}

	public static float[] toFloatArray(ArrayList<Float> x) {
		float[] result = new float[x.size()];
		for (int i = 0; i < x.size(); i++) {
			result[i] = x.get(i);
		}
		return result;
	}

	public static int[] toIntArray(ArrayList<Integer> x) {
		int[] result = new int[x.size()];
		for (int i = 0; i < x.size(); i++) {
			result[i] = x.get(i);
		}
		return result;
	}

	
	public static float[][] toFloatArray2d(ArrayList<float[]> x) {
		float[][] result = new float[x.size()][];
		for (int i = 0; i < x.size(); i++) {
			result[i] = new float[x.get(i).length];
			for (int j = 0; j < x.get(i).length; j++) {
				result[i][j] = x.get(i)[j];
			}
		}
		return result;
	}

	public static float[] toFloatArray(double[] x) {
		float[] result = new float[x.length];
		for (int i = 0; i < x.length; i++) {
			result[i] = (float) x[i];
		}
		return result;
	}

	public static float[] ewAdd(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new RuntimeException("For elementwise operations arrays should be equal-sized");
		}
		float[] result = new float[a.length];
		for (int i = 0; i < a.length; i++) {
			result[i] = a[i] + b[i];
		}
		return result;
	}

	public static float[] absDelta(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new RuntimeException("For elementwise operations arrays should be equal-sized");
		}
		float[] result = new float[a.length];
		for (int i = 0; i < a.length; i++) {
			result[i] = Math.abs(a[i] - b[i]);
		}
		return result;
	}

	public static float[] ewMult(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new RuntimeException("For elementwise operations arrays should be equal-sized");
		}
		float[] result = new float[a.length];
		for (int i = 0; i < a.length; i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}

	public static float[] ewDiv(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new RuntimeException("For elementwise operations arrays should be equal-sized");
		}
		float[] result = new float[a.length];
		for (int i = 0; i < a.length; i++) {
			result[i] = a[i] / b[i];
		}
		return result;
	}

	public static float[] mult(float a, float[] b) {
		float[] result = new float[b.length];
		for (int i = 0; i < b.length; i++) {
			result[i] = a * b[i];
		}
		return result;
	}

	public static float[] abs(float[] a) {
		float[] result = new float[a.length];
		for (int i = 0; i < a.length; i++) {
			result[i] = Math.abs(a[i]);
		}
		return result;
	}

	/**
	 * 
	 * @param a array of floats
	 * @return average value of elements of a. If a consists of 1.0F, 2.0F, 6.0F
	 *         result will be 3.0F
	 */
	public static float mean(float[] a) {
		float m = 0;
		for (int i = 0; i < a.length; i++) {
			m += a[i];
		}
		return m / ((float) a.length);
	}

	/**
	 * 1.0 for two perfectly constant arrays, zero for constant and non-constant
	 * 
	 * @param a float array
	 * @param b float array
	 * @return correlation (Pearson)
	 */
	public static float corr(float[] a, float[] b) {
		return (float) Math.sqrt(r2(a, b));
	}

	/**
	 * 1.0 for two perfectly constant arrays, zero for constant and non-constant
	 * 
	 * @param a float array
	 * @param b float array
	 * @return coefficient of determination
	 */
	public static float r2(float[] a, float[] b) {
		float ma = mean(a);
		float mb = mean(b);
		float s1 = 0;
		float s2 = 0;
		float s3 = 0;
		for (int i = 0; i < a.length; i++) {
			s1 += (a[i] - ma) * (b[i] - mb);
			s2 += (a[i] - ma) * (a[i] - ma);
			s3 += (b[i] - mb) * (b[i] - mb);
		}
		if (s2 * s3 == 0) {
			return ((s2 == 0) && (s3 == 0)) ? 1.0f : 0.0f;
		}
		return (s1 * s1) / (s2 * s3);
	}

	/**
	 * 
	 * @param a array of floats
	 * @return median value of elements of a. If a consists of 1.0F, 2.0F, 6.0F
	 *         result will be 2.0F
	 */
	public static float median(float[] a) {
		if (a.length == 0) {
			return (Float.NaN);
		}
		float[] b = a.clone();
		Arrays.sort(b);
		if (a.length % 2 == 1) {
			int n = a.length / 2;
			return b[n];
		} else {
			int n = a.length / 2;
			return ((b[n] + b[n - 1]) / 2.0F);
		}
	}

	/**
	 * 
	 * @param a float array
	 * @param f - fraction of array elements
	 * @return such value, that f of array elements are lower than it
	 */
	public static float deltaLowerF(float[] a, float f) {
		if (a.length == 0) {
			return (Float.NaN);
		}
		float[] b = a.clone();
		Arrays.sort(b);
		int n = Math.round(a.length * f);
		return b[n];
	}

	/**
	 * 
	 * @param a float array
	 * @return such value, that 90% of array elements are lower than it
	 */
	public static float delta90(float[] a) {
		return deltaLowerF(a, 0.9f);
	}

	/**
	 * 
	 * @param a float array
	 * @return such value, that 80% of array elements are lower than it
	 */
	public static float delta80(float[] a) {
		return deltaLowerF(a, 0.8f);
	}

	/**
	 * 
	 * @param a float array
	 * @return such value, that 95% of array elements are lower than it
	 */
	public static float delta95(float[] a) {
		return deltaLowerF(a, 0.95f);
	}

	public static void savePredResults(FileWriter fw, String[] strings, float[] val1, float[] val2) throws IOException {
		if (strings.length != val1.length) {
			throw new RuntimeException("Different lengths of arrays");
		}
		if (strings.length != val2.length) {
			throw new RuntimeException("Different lengths of arrays");
		}
		for (int i = 0; i < strings.length; i++) {
			fw.write(strings[i] + " " + val1[i] + " " + val2[i] + "\n");
		}
	}

	public static float[] randomArray(float[] min, float[] max) {
		if (min.length != max.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		float[] result = new float[min.length];
		for (int i = 0; i < min.length; i++) {
			result[i] = min[i] + ((float) Math.random()) * (max[i] - min[i]);
		}
		return result;
	}

	public static String valuesAndNamesToString(String[] valNames, float[] values) {
		if (valNames.length != values.length) {
			throw (new RuntimeException("Wrong length of arrays" + " " + valNames.length + " " + values.length));
		}
		String result = "";
		for (int i = 0; i < valNames.length; i++) {
			result += valNames[i] + " " + values[i] + " ";
		}
		return result;
	}

	/**
	 * 
	 * @param a array of float
	 * @param b array of float
	 * @return concatenate two arrays. float[a.length+b.length]
	 */
	public static float[] mergeArrays(float a[], float b[]) {
		float[] result = new float[a.length + b.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (i < a.length) ? a[i] : b[i - a.length];
		}
		return result;
	}

	/**
	 * 
	 * @param a array of byte
	 * @param b array of byte
	 * @return concatenate two arrays. float[a.length+b.length]
	 */
	public static byte[] mergeArrays(byte a[], byte b[]) {
		byte[] result = new byte[a.length + b.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (i < a.length) ? a[i] : b[i - a.length];
		}
		return result;
	}
	
	/**
	 * 
	 * @param a array of String
	 * @param b array of String
	 * @return concatenate two arrays. String[a.length+b.length]
	 */
	public static String[] mergeArrays(String a[], String b[]) {
		String[] result = new String[a.length + b.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (i < a.length) ? a[i] : b[i - a.length];
		}
		return result;
	}

	public static float[] splitArrayBeginning(float a[], int lengthFirst) {
		float[] result = new float[lengthFirst];
		if (a.length < result.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		for (int i = 0; i < result.length; i++) {
			result[i] = a[i];
		}
		return result;
	}

	public static float[] splitArrayEnding(float a[], int lengthFirst) {
		float[] result = new float[a.length - lengthFirst];
		if (a.length < lengthFirst) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		for (int i = lengthFirst; i < a.length; i++) {
			result[i - lengthFirst] = a[i];
		}
		return result;
	}

	public static int[] ints(int n) {
		int[] a = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = i;
		}
		return a;
	}

	public static int[] intsrnd(int n) {
		int[] a = ints(n);
		Random rnd = new Random();
		for (int i = 0; i < n; i++) {
			int x = a[i];
			int j = rnd.nextInt(n);
			int y = a[j];
			a[i] = y;
			a[j] = x;
		}
		return a;
	}

	public static String[] intStrings(int n) {
		String[] a = new String[n];
		for (int i = 0; i < n; i++) {
			a[i] = i + "";
		}
		return a;
	}

	public static DataFrame toDataFrame(float[][] features, float[] labels) {
		double[][] featuresDouble = toDoubleArray2d(features);
		double[][] labelsDouble = toDoubleArray2d(labels);
		String[] names = intStrings(features[0].length);
		if (features.length != labels.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		DataFrame dataFrame = DataFrame.of(labelsDouble, "label").merge(DataFrame.of(featuresDouble, names));
		return dataFrame;
	}

	public static DataFrame toDataFrame(float[][] features) {
		return toDataFrame(features, new float[features.length]);
	}

	public static DataFrame toDataFrame(float[] features) {
		return toDataFrame(new float[][] { features });
	}

	public static DataFrame toDataFrame(float[][] features, StructType schema) {

		DataFrame dataFrame = toDataFrame(features);
		dataFrame = DataFrame.of(dataFrame.stream(), schema);
		return dataFrame;
	}

	public static DataFrame toDataFrame(float[] features, StructType schema) {
		return toDataFrame(new float[][] { features }, schema);
	}

	public static float scaleTo01(float val, float min, float max) {
		if (max - min == 0) {
			return 0f;
		}
		return ((val - min) / (max - min));
	}
	
	public static float[] scaleTo01(float[] arr, float[] min, float[] max) {
		if (arr.length != min.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		if (arr.length != max.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		float[] result = new float[min.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = scaleTo01(arr[i], min[i], max[i]);
		}
		return result;
	}

	public static float[] scaleTo01(float[] a) {
		float min = Float.MAX_VALUE;
		float max = Float.MIN_VALUE;
		for (int i = 0; i < a.length; i++) {
			min = a[i] < min ? a[i] : min;
			max = a[i] > max ? a[i] : max;
		}
		float[] r = new float[a.length];
		for (int i = 0; i < r.length; i++) {
			r[i] = scaleTo01(a[i], min, max);
		}
		return r;
	}

	public static Pair<Float, Float> minmax(float[] a) {
		float min = Float.MAX_VALUE;
		float max = Float.MIN_VALUE;
		for (int i = 0; i < a.length; i++) {
			min = a[i] < min ? a[i] : min;
			max = a[i] > max ? a[i] : max;
		}
		return Pair.of(min, max);
	}

	public static float[] scaleTo01(float[] arr, Pair<float[], float[]> minmax) {
		float[] min = minmax.getLeft();
		float[] max = minmax.getRight();
		return scaleTo01(arr, min, max);
	}

	public static float[][] scaleAllTo01(float[][] arr, Pair<float[], float[]> minmax) {
		float[][] rslt = new float[arr.length][];
		for (int i = 0; i < arr.length; i++) {
			rslt[i] = scaleTo01(arr[i], minmax);
		}
		return rslt;
	}

	public static float[] scaleAllTo01(float[] arr, float min, float max) {
		float[] rslt = new float[arr.length];
		for (int i = 0; i < arr.length; i++) {
			rslt[i] = scaleTo01(arr[i], min, max);
		}
		return rslt;
	}

	public static float[] scaleBackFrom01All(float[] arr, float min, float max) {
		float[] rslt = new float[arr.length];
		for (int i = 0; i < arr.length; i++) {
			rslt[i] = min + arr[i] * (max - min);
		}
		return rslt;
	}

	public static float[][] scaleAllTo01(float[][] arr, float[] min, float[] max) {
		float[][] rslt = new float[arr.length][];
		for (int i = 0; i < arr.length; i++) {
			rslt[i] = scaleTo01(arr[i], min, max);
		}
		return rslt;
	}

	public abstract static class RandomArrays {
		public abstract float[] randomArray();
	}

	public static float[][] transpose(float[][] a) {
		float[][] result = new float[a[0].length][a.length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[0].length; j++) {
				result[i][j] = a[j][i];
			}
		}
		return result;
	}

	public static byte[][] transpose(byte[][] a) {
		byte[][] result = new byte[a[0].length][a.length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[0].length; j++) {
				result[i][j] = a[j][i];
			}
		}
		return result;
	}
	
	public static Pair<float[][], float[]> randomSubset(float[][] features, float[] labels, float fraction) {
		if (features.length != labels.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		boolean[] a = new boolean[labels.length];
		int n = 0;
		for (int i = 0; i < labels.length; i++) {
			a[i] = Math.random() < fraction;
			n = n + (a[i] ? 1 : 0);
		}
		float[][] result1 = new float[n][];
		float[] result2 = new float[n];
		int j = 0;
		for (int i = 0; i < labels.length; i++) {
			if (a[i]) {
				result1[j] = features[j];
				result2[j] = labels[j];
				j++;
			}
		}
		if (j != n) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		return Pair.of(result1, result2);
	}

	public static DMatrix dataSetToXGBooostDMatrix(float[][] features, float[] labels) throws IOException {
		FileWriter f = new FileWriter("./XGBoost.tmp");
		for (int i = 0; i < labels.length; i++) {
			f.write(labels[i] + " ");
			for (int j = 0; j < features[i].length; j++) {
				f.write(j + ":" + features[i][j] + " ");
			}
			f.write("\n");
		}
		f.close();
		try {
			DMatrix mat = new DMatrix("./XGBoost.tmp");
			(new File("./XGBoost.tmp")).delete();
			return mat;
		} catch (XGBoostError e) {
			throw new IOException(e.getMessage());
		}
	}

	public static float[] flatten(float[][] a) {
		float[] result = new float[a.length * a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				result[a[0].length * i + j] = a[i][j];
			}
		}
		return result;
	}

	public static float distance(float[] a, float[] b, boolean manhattan) {
		if (a.length != b.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		float sum = 0;
		for (int i = 0; i < a.length; i++) {
			float d = Math.abs(a[i] - b[i]);
			sum += manhattan ? d : d * d;
		}
		sum = manhattan ? sum : (float) Math.sqrt(sum);
		return sum;
	}

	public static int[] kLeastElements(float[] a, int k) {
		if (k > a.length) {
			throw (new RuntimeException("Wrong length of arrays"));
		}
		int[] result = new int[k];
		for (int i = 0; i < k; i++) {
			result[i] = -1;
		}
		for (int i = 0; i < k; i++) {
			float min = Float.POSITIVE_INFINITY;
			int bestJ = -1;
			for (int j = 0; j < a.length; j++) {
				if (a[j] < min) {
					boolean yetFound = false;
					for (int x = 0; x < i; x++) {
						if (result[x] == j) {
							yetFound = true;
						}
					}
					if (!yetFound) {
						min = a[j];
						bestJ = j;
					}
				}
			}
			result[i] = bestJ;
		}
		return result;
	}

	public static int[] knn(float[][] set, float[] query, int k, boolean manhattan) {
		float[] distances = new float[set.length];
		for (int i = 0; i < distances.length; i++) {
			distances[i] = distance(set[i], query, manhattan);
		}
		return kLeastElements(distances, k);
	}

	public static svm_node[] toLibSVMFormat(float[] features) {
		svm_node[] x = new svm_node[features.length];
		for (int i = 0; i < x.length; i++) {
			x[i] = new svm_node();
			x[i].index = i + 1;
			x[i].value = features[i];
		}
		return x;
	}

	public static svm_node[][] toLibSVMFormat(float[][] features) {
		svm_node[][] x = new svm_node[features.length][];
		for (int i = 0; i < x.length; i++) {
			x[i] = toLibSVMFormat(features[i]);
		}
		return x;
	}

	public static svm_problem toLibSVMFormat(float[][] features, float[] labels) {
		svm_problem p = new svm_problem();
		p.y = toDoubleArray(labels);
		p.x = toLibSVMFormat(features);
		p.l = labels.length;
		return p;
	}

	public static Feature[] toLibLinearFormat(float[] features) {
		Feature[] x = new Feature[features.length];
		for (int i = 0; i < x.length; i++) {
			x[i] = new FeatureNode(i + 1, features[i]);
		}
		return x;
	}

	public static Feature[][] toLibLinearFormat(float[][] features) {
		Feature[][] x = new Feature[features.length][];
		for (int i = 0; i < x.length; i++) {
			x[i] = toLibLinearFormat(features[i]);
		}
		return x;
	}

	public static Problem toLibLinearFormat(float[][] features, float[] labels) {
		Problem p = new Problem();
		p.y = toDoubleArray(labels);
		p.x = toLibLinearFormat(features);
		p.l = labels.length;
		p.n = features[0].length;
		return p;
	}
}
